#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "VexisPresenceBridgeSubsystem.h"
#include "VexisPresenceActor.generated.h"

class UAnimationAsset;
class UCameraComponent;
class UPointLightComponent;
class USceneCaptureComponent2D;
class USceneComponent;
class USkeletalMeshComponent;
class UTextureRenderTarget2D;

UCLASS()
class VEXISPRESENCE_API AVexisPresenceActor : public AActor
{
	GENERATED_BODY()

public:
	AVexisPresenceActor();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable, Category = "VEXIS")
	void ApplyPresenceSnapshot(const FVexisPresenceSnapshot& Snapshot);

	UFUNCTION(BlueprintPure, Category = "VEXIS")
	UTextureRenderTarget2D* GetOverlayRenderTarget() const;

	UFUNCTION(BlueprintPure, Category = "VEXIS")
	UTextureRenderTarget2D* GetOverlayAlphaRenderTarget() const;

	UFUNCTION(BlueprintCallable, Category = "VEXIS")
	void CaptureOverlayFrame();

protected:
	UFUNCTION()
	void HandlePresenceSnapshotUpdated(const FVexisPresenceSnapshot& Snapshot);

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<USceneComponent> SceneRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<USceneComponent> AvatarRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<USkeletalMeshComponent> AvatarMesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<UPointLightComponent> KeyLight;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<UPointLightComponent> FillLight;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<UCameraComponent> ViewCamera;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<USceneCaptureComponent2D> OverlayColorCapture;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VEXIS")
	TObjectPtr<USceneCaptureComponent2D> OverlayAlphaCapture;

private:
	FVexisPresenceSnapshot LastSnapshot;
	FVector BaseLocation = FVector::ZeroVector;
	FRotator BaseRotation = FRotator::ZeroRotator;
	float RunningTime = 0.0f;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> OverlayRenderTarget;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> OverlayAlphaRenderTarget;

	UPROPERTY(Transient)
	TObjectPtr<UAnimationAsset> IdleAnimationAsset;
};
