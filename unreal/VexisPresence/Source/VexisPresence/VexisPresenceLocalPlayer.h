#pragma once

#include "CoreMinimal.h"
#include "Engine/LocalPlayer.h"
#include "VexisPresenceLocalPlayer.generated.h"

UCLASS()
class VEXISPRESENCE_API UVexisPresenceLocalPlayer : public ULocalPlayer
{
	GENERATED_BODY()

public:
	virtual bool CalcSceneViewInitOptions(
		struct FSceneViewInitOptions& OutInitOptions,
		FViewport* Viewport,
		class FViewElementDrawer* ViewDrawer = nullptr,
		int32 StereoViewIndex = INDEX_NONE) override;
};
