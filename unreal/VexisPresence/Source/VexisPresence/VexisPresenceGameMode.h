#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "VexisPresenceGameMode.generated.h"

class AVexisPresenceActor;

UCLASS()
class VEXISPRESENCE_API AVexisPresenceGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AVexisPresenceGameMode();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	UPROPERTY(EditDefaultsOnly, Category = "VEXIS")
	TSubclassOf<AVexisPresenceActor> PresenceActorClass;

private:
	UFUNCTION()
	void CreateOverlayPresenceWindow();

	UFUNCTION()
	void UpdateOverlayPresenceWindow();

	void HidePrimaryGameWindow();
	void DestroyOverlayPresenceWindow();
	bool EnsureOverlayBitmapResources(int32 Width, int32 Height);
	void SyncOverlayWindowInteraction();

	TObjectPtr<AVexisPresenceActor> SpawnedPresenceActor;
	FTimerHandle OverlayRefreshTimerHandle;
	void* OverlayWindowHandle = nullptr;
	void* OverlayMemoryDeviceContext = nullptr;
	void* OverlayBitmapHandle = nullptr;
	void* OverlayBitmapBits = nullptr;
	int32 OverlayBitmapWidth = 0;
	int32 OverlayBitmapHeight = 0;
	int32 OverlayWindowX = 70;
	int32 OverlayWindowY = 110;
	bool bOverlayWindowCreated = false;
	bool bOverlayDragModeActive = false;
};
