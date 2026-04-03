#include "VexisPresenceLocalPlayer.h"

#include "SceneView.h"

bool UVexisPresenceLocalPlayer::CalcSceneViewInitOptions(
	FSceneViewInitOptions& OutInitOptions,
	FViewport* Viewport,
	FViewElementDrawer* ViewDrawer,
	int32 StereoViewIndex)
{
	if (!Super::CalcSceneViewInitOptions(OutInitOptions, Viewport, ViewDrawer, StereoViewIndex))
	{
		return false;
	}

	OutInitOptions.BackgroundColor = FLinearColor(0.0f, 0.0f, 0.0f, 0.0f);
	return true;
}
