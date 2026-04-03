#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Tickable.h"
#include "Interfaces/IHttpRequest.h"
#include "VexisPresenceBridgeSubsystem.generated.h"

USTRUCT(BlueprintType)
struct FVexisPresenceSnapshot
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	FString Timestamp;

	UPROPERTY(BlueprintReadOnly)
	FString VisualState = TEXT("idle");

	UPROPERTY(BlueprintReadOnly)
	FString StatusText = TEXT("ready");

	UPROPERTY(BlueprintReadOnly)
	bool bIsThinking = false;

	UPROPERTY(BlueprintReadOnly)
	bool bIsSpeaking = false;

	UPROPERTY(BlueprintReadOnly)
	bool bBootInProgress = false;

	UPROPERTY(BlueprintReadOnly)
	FString CurrentFocus;

	UPROPERTY(BlueprintReadOnly)
	FString LastClassification;

	UPROPERTY(BlueprintReadOnly)
	FString ActivePresenceAction;

	UPROPERTY(BlueprintReadOnly)
	FString LastResponseText;

	UPROPERTY(BlueprintReadOnly)
	FString LastSpokenText;

	UPROPERTY(BlueprintReadOnly)
	TArray<FString> ThoughtLines;

	UPROPERTY(BlueprintReadOnly)
	int32 OpenQuestionCount = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 UnsupportedClaimCount = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 CompletedFileCount = 0;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVexisPresenceSnapshotUpdated, const FVexisPresenceSnapshot&, Snapshot);

UCLASS(BlueprintType)
class VEXISPRESENCE_API UVexisPresenceBridgeSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override;
	virtual bool IsTickable() const override;
	virtual bool IsTickableInEditor() const override;

	UFUNCTION(BlueprintPure, Category = "VEXIS")
	const FVexisPresenceSnapshot& GetCurrentSnapshot() const;

	UFUNCTION(BlueprintCallable, Category = "VEXIS")
	void ForceRefresh();

	UPROPERTY(BlueprintAssignable, Category = "VEXIS")
	FVexisPresenceSnapshotUpdated OnPresenceSnapshotUpdated;

private:
	void RequestSnapshot();
	void HandleSnapshotResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful);
	bool ParseSnapshotJson(const FString& JsonText, FVexisPresenceSnapshot& OutSnapshot) const;
	void LoadConfig();
	void HandleBridgeFailure();

	FVexisPresenceSnapshot CurrentSnapshot;
	float PollIntervalSeconds = 0.25f;
	float TimeSinceLastPoll = 0.0f;
	bool bRequestInFlight = false;
	FString EndpointUrl = TEXT("http://127.0.0.1:8765/v1/presence");
	int32 ConsecutiveFailureCount = 0;
	bool bHasReceivedSnapshot = false;
	float TimeSinceLastSuccessfulSnapshot = 0.0f;
	int32 FailureExitThreshold = 2;
	float InitialGracePeriodSeconds = 2.5f;
};
