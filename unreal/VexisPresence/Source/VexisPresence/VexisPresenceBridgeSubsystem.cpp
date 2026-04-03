#include "VexisPresenceBridgeSubsystem.h"

#include "Dom/JsonObject.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Misc/ConfigCacheIni.h"
#include "GenericPlatform/GenericPlatformMisc.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

void UVexisPresenceBridgeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	LoadConfig();
	TimeSinceLastPoll = PollIntervalSeconds;
}

void UVexisPresenceBridgeSubsystem::Deinitialize()
{
	bRequestInFlight = false;
	Super::Deinitialize();
}

void UVexisPresenceBridgeSubsystem::Tick(float DeltaTime)
{
	TimeSinceLastPoll += DeltaTime;
	TimeSinceLastSuccessfulSnapshot += DeltaTime;

	if (!bRequestInFlight && TimeSinceLastPoll >= PollIntervalSeconds)
	{
		RequestSnapshot();
	}

	if (!bHasReceivedSnapshot && TimeSinceLastSuccessfulSnapshot >= InitialGracePeriodSeconds)
	{
		FPlatformMisc::RequestExit(false, TEXT("VexisPresenceBridgeSubsystem.InitialBridgeTimeout"));
	}
}

TStatId UVexisPresenceBridgeSubsystem::GetStatId() const
{
	RETURN_QUICK_DECLARE_CYCLE_STAT(UVexisPresenceBridgeSubsystem, STATGROUP_Tickables);
}

bool UVexisPresenceBridgeSubsystem::IsTickable() const
{
	return GetGameInstance() != nullptr;
}

bool UVexisPresenceBridgeSubsystem::IsTickableInEditor() const
{
	return true;
}

const FVexisPresenceSnapshot& UVexisPresenceBridgeSubsystem::GetCurrentSnapshot() const
{
	return CurrentSnapshot;
}

void UVexisPresenceBridgeSubsystem::ForceRefresh()
{
	if (!bRequestInFlight)
	{
		RequestSnapshot();
	}
}

void UVexisPresenceBridgeSubsystem::LoadConfig()
{
	if (GConfig == nullptr)
	{
		return;
	}

	FString ConfiguredEndpoint;
	if (GConfig->GetString(TEXT("/Script/VexisPresence.VexisPresenceRuntimeSettings"), TEXT("BridgeEndpoint"), ConfiguredEndpoint, GGameIni) && !ConfiguredEndpoint.IsEmpty())
	{
		EndpointUrl = ConfiguredEndpoint;
	}

	float ConfiguredPollInterval = PollIntervalSeconds;
	if (GConfig->GetFloat(TEXT("/Script/VexisPresence.VexisPresenceRuntimeSettings"), TEXT("PollIntervalSeconds"), ConfiguredPollInterval, GGameIni))
	{
		PollIntervalSeconds = FMath::Max(0.05f, ConfiguredPollInterval);
	}
}

void UVexisPresenceBridgeSubsystem::RequestSnapshot()
{
	TimeSinceLastPoll = 0.0f;
	bRequestInFlight = true;

	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
	Request->SetVerb(TEXT("GET"));
	Request->SetURL(EndpointUrl);
	Request->SetHeader(TEXT("Accept"), TEXT("application/json"));
	Request->SetTimeout(0.75f);
	Request->SetActivityTimeout(0.75f);
	Request->OnProcessRequestComplete().BindUObject(this, &UVexisPresenceBridgeSubsystem::HandleSnapshotResponse);
	Request->ProcessRequest();
}

void UVexisPresenceBridgeSubsystem::HandleSnapshotResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful)
{
	bRequestInFlight = false;

	if (!bWasSuccessful || !Response.IsValid())
	{
		HandleBridgeFailure();
		return;
	}

	if (Response->GetResponseCode() != 200)
	{
		HandleBridgeFailure();
		return;
	}

	FVexisPresenceSnapshot ParsedSnapshot;
	if (!ParseSnapshotJson(Response->GetContentAsString(), ParsedSnapshot))
	{
		HandleBridgeFailure();
		return;
	}

	ConsecutiveFailureCount = 0;
	bHasReceivedSnapshot = true;
	TimeSinceLastSuccessfulSnapshot = 0.0f;
	CurrentSnapshot = ParsedSnapshot;
	OnPresenceSnapshotUpdated.Broadcast(CurrentSnapshot);
}

bool UVexisPresenceBridgeSubsystem::ParseSnapshotJson(const FString& JsonText, FVexisPresenceSnapshot& OutSnapshot) const
{
	TSharedPtr<FJsonObject> JsonObject;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
	if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
	{
		return false;
	}

	JsonObject->TryGetStringField(TEXT("timestamp"), OutSnapshot.Timestamp);
	JsonObject->TryGetStringField(TEXT("visual_state"), OutSnapshot.VisualState);
	JsonObject->TryGetStringField(TEXT("status_text"), OutSnapshot.StatusText);
	JsonObject->TryGetBoolField(TEXT("is_thinking"), OutSnapshot.bIsThinking);
	JsonObject->TryGetBoolField(TEXT("is_speaking"), OutSnapshot.bIsSpeaking);
	JsonObject->TryGetBoolField(TEXT("boot_in_progress"), OutSnapshot.bBootInProgress);
	JsonObject->TryGetStringField(TEXT("current_focus"), OutSnapshot.CurrentFocus);
	JsonObject->TryGetStringField(TEXT("last_classification"), OutSnapshot.LastClassification);
	JsonObject->TryGetStringField(TEXT("active_presence_action"), OutSnapshot.ActivePresenceAction);
	JsonObject->TryGetStringField(TEXT("last_response_text"), OutSnapshot.LastResponseText);
	JsonObject->TryGetStringField(TEXT("last_spoken_text"), OutSnapshot.LastSpokenText);
	JsonObject->TryGetNumberField(TEXT("open_question_count"), OutSnapshot.OpenQuestionCount);
	JsonObject->TryGetNumberField(TEXT("unsupported_claim_count"), OutSnapshot.UnsupportedClaimCount);
	JsonObject->TryGetNumberField(TEXT("completed_file_count"), OutSnapshot.CompletedFileCount);

	OutSnapshot.ThoughtLines.Reset();
	const TArray<TSharedPtr<FJsonValue>>* ThoughtArray = nullptr;
	if (JsonObject->TryGetArrayField(TEXT("thought_lines"), ThoughtArray) && ThoughtArray != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *ThoughtArray)
		{
			FString Line;
			if (Value.IsValid() && Value->TryGetString(Line))
			{
				OutSnapshot.ThoughtLines.Add(Line);
			}
		}
	}

	return true;
}

void UVexisPresenceBridgeSubsystem::HandleBridgeFailure()
{
	ConsecutiveFailureCount += 1;
	if (bHasReceivedSnapshot && ConsecutiveFailureCount >= FailureExitThreshold)
	{
		FPlatformMisc::RequestExit(false, TEXT("VexisPresenceBridgeSubsystem.BridgeLost"));
	}
}
