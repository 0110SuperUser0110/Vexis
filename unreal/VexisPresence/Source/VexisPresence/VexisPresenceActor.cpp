#include "VexisPresenceActor.h"

#include "Animation/AnimationAsset.h"
#include "Camera/CameraComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Materials/MaterialInterface.h"
#include "UObject/ConstructorHelpers.h"

namespace
{
	constexpr int32 VexisOverlayRenderSize = 640;
}

AVexisPresenceActor::AVexisPresenceActor()
{
	PrimaryActorTick.bCanEverTick = true;

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SetRootComponent(SceneRoot);

	AvatarRoot = CreateDefaultSubobject<USceneComponent>(TEXT("AvatarRoot"));
	AvatarRoot->SetupAttachment(SceneRoot);
	AvatarRoot->SetRelativeLocation(FVector(0.0f, 0.0f, -8.0f));

	AvatarMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("AvatarMesh"));
	AvatarMesh->SetupAttachment(AvatarRoot);
	AvatarMesh->SetMobility(EComponentMobility::Movable);
	AvatarMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	AvatarMesh->SetGenerateOverlapEvents(false);
	AvatarMesh->SetCastShadow(false);
	AvatarMesh->bCastDynamicShadow = false;
	AvatarMesh->SetRelativeLocation(FVector(0.0f, 0.0f, -122.0f));
	AvatarMesh->SetRelativeRotation(FRotator(0.0f, -90.0f, 0.0f));
	AvatarMesh->SetRelativeScale3D(FVector(0.90f, 0.90f, 0.90f));
	AvatarMesh->SetAnimationMode(EAnimationMode::AnimationSingleNode);

	static ConstructorHelpers::FObjectFinder<USkeletalMesh> QuinnMesh(
		TEXT("/Game/CombatMagicAnims/Demo/Mannequins/Meshes/SKM_Quinn_Simple.SKM_Quinn_Simple")
	);
	if (QuinnMesh.Succeeded())
	{
		AvatarMesh->SetSkeletalMeshAsset(QuinnMesh.Object);
	}

	static ConstructorHelpers::FObjectFinder<UMaterialInterface> QuinnMaterialBody(
		TEXT("/Game/CombatMagicAnims/Demo/Mannequins/Materials/Quinn/MI_Quinn_01.MI_Quinn_01")
	);
	static ConstructorHelpers::FObjectFinder<UMaterialInterface> QuinnMaterialHead(
		TEXT("/Game/CombatMagicAnims/Demo/Mannequins/Materials/Quinn/MI_Quinn_02.MI_Quinn_02")
	);
	if (QuinnMaterialBody.Succeeded())
	{
		AvatarMesh->SetMaterial(0, QuinnMaterialBody.Object);
	}
	if (QuinnMaterialHead.Succeeded())
	{
		AvatarMesh->SetMaterial(1, QuinnMaterialHead.Object);
	}

	static ConstructorHelpers::FObjectFinder<UAnimationAsset> IdleAnim(
		TEXT("/Game/CombatMagicAnims/Demo/Mannequins/Anims/Unarmed/MM_Idle.MM_Idle")
	);
	if (IdleAnim.Succeeded())
	{
		IdleAnimationAsset = IdleAnim.Object;
		AvatarMesh->SetAnimation(IdleAnimationAsset);
	}

	KeyLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("KeyLight"));
	KeyLight->SetupAttachment(SceneRoot);
	KeyLight->SetRelativeLocation(FVector(96.0f, -24.0f, 42.0f));
	KeyLight->SetIntensity(430.0f);
	KeyLight->SetAttenuationRadius(280.0f);
	KeyLight->bUseInverseSquaredFalloff = false;
	KeyLight->LightColor = FColor(228, 226, 224);
	KeyLight->SetCastShadows(false);

	FillLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("FillLight"));
	FillLight->SetupAttachment(SceneRoot);
	FillLight->SetRelativeLocation(FVector(28.0f, 62.0f, 16.0f));
	FillLight->SetIntensity(58.0f);
	FillLight->SetAttenuationRadius(220.0f);
	FillLight->bUseInverseSquaredFalloff = false;
	FillLight->LightColor = FColor(196, 198, 204);
	FillLight->SetCastShadows(false);

	ViewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("ViewCamera"));
	ViewCamera->SetupAttachment(SceneRoot);
	ViewCamera->SetRelativeLocation(FVector(150.0f, 0.0f, 24.0f));
	ViewCamera->SetRelativeRotation(FRotator(0.0f, 180.0f, 0.0f));
	ViewCamera->FieldOfView = 13.5f;
	ViewCamera->PostProcessSettings.bOverride_AutoExposureMethod = true;
	ViewCamera->PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Manual;
	ViewCamera->PostProcessSettings.bOverride_AutoExposureBias = true;
	ViewCamera->PostProcessSettings.AutoExposureBias = -2.75f;
	ViewCamera->PostProcessSettings.bOverride_MotionBlurAmount = true;
	ViewCamera->PostProcessSettings.MotionBlurAmount = 0.0f;

	OverlayColorCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("OverlayColorCapture"));
	OverlayColorCapture->SetupAttachment(SceneRoot);
	OverlayColorCapture->SetRelativeLocation(ViewCamera->GetRelativeLocation());
	OverlayColorCapture->SetRelativeRotation(ViewCamera->GetRelativeRotation());
	OverlayColorCapture->ProjectionType = ECameraProjectionMode::Perspective;
	OverlayColorCapture->FOVAngle = ViewCamera->FieldOfView;
	OverlayColorCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	OverlayColorCapture->CompositeMode = SCCM_Overwrite;
	OverlayColorCapture->bCaptureEveryFrame = false;
	OverlayColorCapture->bCaptureOnMovement = false;
	OverlayColorCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
	OverlayColorCapture->bConsiderUnrenderedOpaquePixelAsFullyTranslucent = true;
	OverlayColorCapture->PostProcessBlendWeight = 1.0f;
	OverlayColorCapture->PostProcessSettings.bOverride_AutoExposureMethod = true;
	OverlayColorCapture->PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Manual;
	OverlayColorCapture->PostProcessSettings.bOverride_AutoExposureBias = true;
	OverlayColorCapture->PostProcessSettings.AutoExposureBias = -2.75f;
	OverlayColorCapture->PostProcessSettings.bOverride_VignetteIntensity = true;
	OverlayColorCapture->PostProcessSettings.VignetteIntensity = 0.0f;
	OverlayColorCapture->ShowFlags.SetAtmosphere(false);
	OverlayColorCapture->ShowFlags.SetFog(false);
	OverlayColorCapture->ShowFlags.SetCloud(false);
	OverlayColorCapture->ShowFlags.SetSkyLighting(false);
	OverlayColorCapture->ShowFlags.SetTemporalAA(false);

	OverlayAlphaCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("OverlayAlphaCapture"));
	OverlayAlphaCapture->SetupAttachment(SceneRoot);
	OverlayAlphaCapture->SetRelativeLocation(ViewCamera->GetRelativeLocation());
	OverlayAlphaCapture->SetRelativeRotation(ViewCamera->GetRelativeRotation());
	OverlayAlphaCapture->ProjectionType = ECameraProjectionMode::Perspective;
	OverlayAlphaCapture->FOVAngle = ViewCamera->FieldOfView;
	OverlayAlphaCapture->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
	OverlayAlphaCapture->CompositeMode = SCCM_Overwrite;
	OverlayAlphaCapture->bCaptureEveryFrame = false;
	OverlayAlphaCapture->bCaptureOnMovement = false;
	OverlayAlphaCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
	OverlayAlphaCapture->bConsiderUnrenderedOpaquePixelAsFullyTranslucent = true;
	OverlayAlphaCapture->PostProcessBlendWeight = 0.0f;
	OverlayAlphaCapture->ShowFlags.SetAtmosphere(false);
	OverlayAlphaCapture->ShowFlags.SetFog(false);
	OverlayAlphaCapture->ShowFlags.SetCloud(false);
	OverlayAlphaCapture->ShowFlags.SetSkyLighting(false);
	OverlayAlphaCapture->ShowFlags.SetLighting(false);
	OverlayAlphaCapture->ShowFlags.SetTemporalAA(false);

	Tags.Add(TEXT("VexisPresence"));
}

void AVexisPresenceActor::BeginPlay()
{
	Super::BeginPlay();

	BaseLocation = AvatarRoot->GetRelativeLocation();
	BaseRotation = AvatarRoot->GetRelativeRotation();

	if (IdleAnimationAsset != nullptr)
	{
		AvatarMesh->PlayAnimation(IdleAnimationAsset, true);
	}

	if (OverlayRenderTarget == nullptr)
	{
		OverlayRenderTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("OverlayRenderTarget"), RF_Transient);
		OverlayRenderTarget->RenderTargetFormat = RTF_RGBA8;
		OverlayRenderTarget->ClearColor = FLinearColor(0.0f, 0.0f, 0.0f, 1.0f);
		OverlayRenderTarget->bAutoGenerateMips = false;
		OverlayRenderTarget->TargetGamma = 1.0f;
		OverlayRenderTarget->InitAutoFormat(VexisOverlayRenderSize, VexisOverlayRenderSize);
		OverlayRenderTarget->UpdateResourceImmediate(true);
	}

	if (OverlayAlphaRenderTarget == nullptr)
	{
		OverlayAlphaRenderTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("OverlayAlphaRenderTarget"), RF_Transient);
		OverlayAlphaRenderTarget->RenderTargetFormat = RTF_RGBA8;
		OverlayAlphaRenderTarget->ClearColor = FLinearColor(1.0f, 1.0f, 1.0f, 1.0f);
		OverlayAlphaRenderTarget->bAutoGenerateMips = false;
		OverlayAlphaRenderTarget->TargetGamma = 1.0f;
		OverlayAlphaRenderTarget->InitAutoFormat(VexisOverlayRenderSize, VexisOverlayRenderSize);
		OverlayAlphaRenderTarget->UpdateResourceImmediate(true);
	}

	OverlayColorCapture->TextureTarget = OverlayRenderTarget;
	OverlayColorCapture->ClearShowOnlyComponents();
	OverlayColorCapture->ShowOnlyComponent(AvatarMesh);

	OverlayAlphaCapture->TextureTarget = OverlayAlphaRenderTarget;
	OverlayAlphaCapture->ClearShowOnlyComponents();
	OverlayAlphaCapture->ShowOnlyComponent(AvatarMesh);

	CaptureOverlayFrame();

	if (UGameInstance* GameInstance = GetGameInstance())
	{
		if (UVexisPresenceBridgeSubsystem* BridgeSubsystem = GameInstance->GetSubsystem<UVexisPresenceBridgeSubsystem>())
		{
			BridgeSubsystem->OnPresenceSnapshotUpdated.AddDynamic(this, &AVexisPresenceActor::HandlePresenceSnapshotUpdated);
			ApplyPresenceSnapshot(BridgeSubsystem->GetCurrentSnapshot());
		}
	}
}

void AVexisPresenceActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	RunningTime += DeltaTime;

	const bool bDancing = LastSnapshot.VisualState.Equals(TEXT("dancing"), ESearchCase::IgnoreCase);
	const bool bThinking = LastSnapshot.VisualState.Equals(TEXT("thinking"), ESearchCase::IgnoreCase);
	const bool bSpeaking = LastSnapshot.VisualState.Equals(TEXT("speaking"), ESearchCase::IgnoreCase);

	const float BobAmplitude = bDancing ? 2.4f : (bThinking ? 1.3f : 0.45f);
	const float BobSpeed = bDancing ? 2.4f : (bThinking ? 1.3f : 0.75f);
	const float YawAmplitude = bDancing ? 3.5f : (bSpeaking ? 1.4f : 0.35f);
	const float PitchAmplitude = bDancing ? 1.4f : (bThinking ? 0.7f : 0.25f);
	const float RollAmplitude = bDancing ? 1.0f : 0.18f;
	const float RotationSpeed = bDancing ? 2.2f : (bSpeaking ? 1.3f : 0.7f);

	const FVector NewLocation = BaseLocation + FVector(0.0f, 0.0f, FMath::Sin(RunningTime * BobSpeed) * BobAmplitude);
	const FRotator NewRotation = BaseRotation + FRotator(
		FMath::Sin(RunningTime * RotationSpeed * 0.7f) * PitchAmplitude,
		FMath::Sin(RunningTime * RotationSpeed) * YawAmplitude,
		FMath::Sin(RunningTime * RotationSpeed * 0.9f) * RollAmplitude
	);

	AvatarRoot->SetRelativeLocation(NewLocation);
	AvatarRoot->SetRelativeRotation(NewRotation);
}

void AVexisPresenceActor::ApplyPresenceSnapshot(const FVexisPresenceSnapshot& Snapshot)
{
	LastSnapshot = Snapshot;
}

UTextureRenderTarget2D* AVexisPresenceActor::GetOverlayRenderTarget() const
{
	return OverlayRenderTarget;
}

UTextureRenderTarget2D* AVexisPresenceActor::GetOverlayAlphaRenderTarget() const
{
	return OverlayAlphaRenderTarget;
}

void AVexisPresenceActor::CaptureOverlayFrame()
{
	if (OverlayColorCapture != nullptr)
	{
		OverlayColorCapture->CaptureScene();
	}
	if (OverlayAlphaCapture != nullptr)
	{
		OverlayAlphaCapture->CaptureScene();
	}
}

void AVexisPresenceActor::HandlePresenceSnapshotUpdated(const FVexisPresenceSnapshot& Snapshot)
{
	ApplyPresenceSnapshot(Snapshot);
}
