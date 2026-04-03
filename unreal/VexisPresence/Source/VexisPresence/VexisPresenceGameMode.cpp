#include "VexisPresenceGameMode.h"

#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Engine/TextureRenderTarget2D.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "Kismet/GameplayStatics.h"
#include "TimerManager.h"
#include "UnrealClient.h"
#include "VexisPresenceActor.h"

#if PLATFORM_WINDOWS
#include "Windows/AllowWindowsPlatformTypes.h"
#include <windows.h>
#include "Windows/HideWindowsPlatformTypes.h"
#endif

namespace
{
	constexpr int32 VexisOverlayStartX = 70;
	constexpr int32 VexisOverlayStartY = 110;
	constexpr int32 VexisOverlayWidth = 640;
	constexpr int32 VexisOverlayHeight = 640;
	constexpr float VexisOverlayRefreshSeconds = 1.0f / 18.0f;
	constexpr float VexisOverlayBrightnessScale = 0.74f;
	constexpr float VexisOverlayDesaturation = 0.14f;
	constexpr float VexisOverlayRedScale = 0.94f;
	constexpr float VexisOverlayBlueScale = 1.02f;
	constexpr uint8 VexisDepthBackgroundCutoff = 250;
	constexpr uint8 VexisVisibleAlphaFloor = 18;

#if PLATFORM_WINDOWS
	constexpr wchar_t VexisOverlayWindowClassName[] = L"VexisPresenceNativeOverlay";
	ATOM GVexisOverlayWindowClassAtom = 0;

	bool IsOverlayDragModifierPressed()
	{
		return (GetAsyncKeyState(VK_MENU) & 0x8000) != 0 && (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
	}

	uint8 ClampByteFromFloat(float Value)
	{
		return static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(Value), 0, 255));
	}

	LRESULT CALLBACK VexisPresenceOverlayWindowProc(HWND WindowHandle, UINT Message, WPARAM WParam, LPARAM LParam)
	{
		switch (Message)
		{
		case WM_NCHITTEST:
			if (IsOverlayDragModifierPressed())
			{
				return HTCAPTION;
			}
			break;
		case WM_ERASEBKGND:
			return 1;
		default:
			break;
		}

		return DefWindowProc(WindowHandle, Message, WParam, LParam);
	}

	void EnsureVexisOverlayWindowClassRegistered()
	{
		if (GVexisOverlayWindowClassAtom != 0)
		{
			return;
		}

		WNDCLASSW WindowClass = {};
		WindowClass.lpfnWndProc = VexisPresenceOverlayWindowProc;
		WindowClass.hInstance = GetModuleHandle(nullptr);
		WindowClass.lpszClassName = VexisOverlayWindowClassName;
		WindowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
		GVexisOverlayWindowClassAtom = RegisterClassW(&WindowClass);
	}
#endif
}

AVexisPresenceGameMode::AVexisPresenceGameMode()
{
	PresenceActorClass = AVexisPresenceActor::StaticClass();
	DefaultPawnClass = nullptr;
	HUDClass = nullptr;
	OverlayWindowX = VexisOverlayStartX;
	OverlayWindowY = VexisOverlayStartY;
}

void AVexisPresenceGameMode::BeginPlay()
{
	Super::BeginPlay();

	if (GetWorld() == nullptr || PresenceActorClass == nullptr)
	{
		return;
	}

	UGameplayStatics::SetForceDisableSplitscreen(this, true);

	if (SpawnedPresenceActor == nullptr)
	{
		for (TActorIterator<AVexisPresenceActor> It(GetWorld()); It; ++It)
		{
			SpawnedPresenceActor = *It;
			break;
		}
	}

	if (SpawnedPresenceActor == nullptr)
	{
		FActorSpawnParameters SpawnParams;
		SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		SpawnedPresenceActor = GetWorld()->SpawnActor<AVexisPresenceActor>(PresenceActorClass, FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
	}

	if (GEngine != nullptr && GEngine->GameViewport != nullptr)
	{
		GEngine->GameViewport->SetMouseCaptureMode(EMouseCaptureMode::NoCapture);
		GEngine->GameViewport->SetMouseLockMode(EMouseLockMode::DoNotLock);
	}

	if (APlayerController* PlayerController = UGameplayStatics::GetPlayerController(this, 0))
	{
		PlayerController->bShowMouseCursor = false;
		PlayerController->bEnableClickEvents = false;
		PlayerController->bEnableMouseOverEvents = false;
		PlayerController->SetIgnoreLookInput(true);
		PlayerController->SetIgnoreMoveInput(true);
		if (SpawnedPresenceActor != nullptr)
		{
			PlayerController->SetViewTarget(SpawnedPresenceActor);
		}
	}

	GetWorldTimerManager().SetTimerForNextTick(this, &AVexisPresenceGameMode::CreateOverlayPresenceWindow);
}

void AVexisPresenceGameMode::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	DestroyOverlayPresenceWindow();
	Super::EndPlay(EndPlayReason);
}

void AVexisPresenceGameMode::CreateOverlayPresenceWindow()
{
	if (bOverlayWindowCreated)
	{
		return;
	}

	if (SpawnedPresenceActor == nullptr || SpawnedPresenceActor->GetOverlayRenderTarget() == nullptr || SpawnedPresenceActor->GetOverlayAlphaRenderTarget() == nullptr)
	{
		GetWorldTimerManager().SetTimerForNextTick(this, &AVexisPresenceGameMode::CreateOverlayPresenceWindow);
		return;
	}

#if PLATFORM_WINDOWS
	EnsureVexisOverlayWindowClassRegistered();
	if (GVexisOverlayWindowClassAtom == 0)
	{
		GetWorldTimerManager().SetTimerForNextTick(this, &AVexisPresenceGameMode::CreateOverlayPresenceWindow);
		return;
	}

	HWND OverlayWindow = CreateWindowExW(
		WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST,
		VexisOverlayWindowClassName,
		L"Vexis Presence Overlay",
		WS_POPUP,
		OverlayWindowX,
		OverlayWindowY,
		VexisOverlayWidth,
		VexisOverlayHeight,
		nullptr,
		nullptr,
		GetModuleHandle(nullptr),
		nullptr
	);

	if (OverlayWindow == nullptr)
	{
		GetWorldTimerManager().SetTimerForNextTick(this, &AVexisPresenceGameMode::CreateOverlayPresenceWindow);
		return;
	}

	OverlayWindowHandle = OverlayWindow;
	ShowWindow(OverlayWindow, SW_SHOWNOACTIVATE);
	UpdateWindow(OverlayWindow);
#endif

	HidePrimaryGameWindow();
	bOverlayWindowCreated = true;
	GetWorldTimerManager().SetTimer(OverlayRefreshTimerHandle, this, &AVexisPresenceGameMode::UpdateOverlayPresenceWindow, VexisOverlayRefreshSeconds, true, 0.0f);
}

void AVexisPresenceGameMode::UpdateOverlayPresenceWindow()
{
	if (!bOverlayWindowCreated || SpawnedPresenceActor == nullptr)
	{
		return;
	}

	UTextureRenderTarget2D* OverlayRenderTarget = SpawnedPresenceActor->GetOverlayRenderTarget();
	UTextureRenderTarget2D* OverlayAlphaRenderTarget = SpawnedPresenceActor->GetOverlayAlphaRenderTarget();
	if (OverlayRenderTarget == nullptr || OverlayAlphaRenderTarget == nullptr)
	{
		return;
	}

	SpawnedPresenceActor->CaptureOverlayFrame();

	FTextureRenderTargetResource* ColorResource = OverlayRenderTarget->GameThread_GetRenderTargetResource();
	FTextureRenderTargetResource* AlphaResource = OverlayAlphaRenderTarget->GameThread_GetRenderTargetResource();
	if (ColorResource == nullptr || AlphaResource == nullptr)
	{
		return;
	}

	const FIntPoint RenderTargetSize = ColorResource->GetSizeXY();
	if (RenderTargetSize.X <= 0 || RenderTargetSize.Y <= 0)
	{
		return;
	}

	if (!EnsureOverlayBitmapResources(RenderTargetSize.X, RenderTargetSize.Y))
	{
		return;
	}

	TArray<FColor> ColorPixels;
	ColorPixels.SetNumUninitialized(RenderTargetSize.X * RenderTargetSize.Y);
	TArray<FColor> AlphaPixels;
	AlphaPixels.SetNumUninitialized(RenderTargetSize.X * RenderTargetSize.Y);

	FReadSurfaceDataFlags ReadFlags(RCM_UNorm);
	ReadFlags.SetLinearToGamma(false);
	if (!ColorResource->ReadPixels(ColorPixels, ReadFlags) || !AlphaResource->ReadPixels(AlphaPixels, ReadFlags))
	{
		return;
	}

#if PLATFORM_WINDOWS
	HWND OverlayWindow = static_cast<HWND>(OverlayWindowHandle);
	HDC MemoryDeviceContext = static_cast<HDC>(OverlayMemoryDeviceContext);
	uint8* BitmapBits = static_cast<uint8*>(OverlayBitmapBits);
	if (OverlayWindow == nullptr || MemoryDeviceContext == nullptr || BitmapBits == nullptr)
	{
		return;
	}

	SyncOverlayWindowInteraction();

	RECT WindowRect = {};
	if (GetWindowRect(OverlayWindow, &WindowRect))
	{
		OverlayWindowX = WindowRect.left;
		OverlayWindowY = WindowRect.top;
	}

	for (int32 Y = 0; Y < RenderTargetSize.Y; ++Y)
	{
		for (int32 X = 0; X < RenderTargetSize.X; ++X)
		{
			const int32 PixelOffset = (Y * RenderTargetSize.X) + X;
			const FColor& ColorPixel = ColorPixels[PixelOffset];
			const FColor& DepthPixel = AlphaPixels[PixelOffset];
			const int32 PixelIndex = PixelOffset * 4;

			const uint8 DepthValue = DepthPixel.R;
			uint8 Alpha = 0;
			if (DepthValue < VexisDepthBackgroundCutoff)
			{
				Alpha = static_cast<uint8>(FMath::Clamp(255 - DepthValue, 0, 255));
				if (Alpha < VexisVisibleAlphaFloor)
				{
					Alpha = 0;
				}
			}

			if (Alpha == 0)
			{
				BitmapBits[PixelIndex + 0] = 0;
				BitmapBits[PixelIndex + 1] = 0;
				BitmapBits[PixelIndex + 2] = 0;
				BitmapBits[PixelIndex + 3] = 0;
				continue;
			}

			float Red = static_cast<float>(ColorPixel.R);
			float Green = static_cast<float>(ColorPixel.G);
			float Blue = static_cast<float>(ColorPixel.B);
			const float Luma = (Red * 0.299f) + (Green * 0.587f) + (Blue * 0.114f);

			Red = FMath::Lerp(Red, Luma, VexisOverlayDesaturation) * VexisOverlayBrightnessScale * VexisOverlayRedScale;
			Green = FMath::Lerp(Green, Luma, VexisOverlayDesaturation) * VexisOverlayBrightnessScale;
			Blue = FMath::Lerp(Blue, Luma, VexisOverlayDesaturation) * VexisOverlayBrightnessScale * VexisOverlayBlueScale;

			const uint8 AdjustedBlue = ClampByteFromFloat(Blue);
			const uint8 AdjustedGreen = ClampByteFromFloat(Green);
			const uint8 AdjustedRed = ClampByteFromFloat(Red);

			BitmapBits[PixelIndex + 0] = static_cast<uint8>((static_cast<uint16>(AdjustedBlue) * Alpha) / 255);
			BitmapBits[PixelIndex + 1] = static_cast<uint8>((static_cast<uint16>(AdjustedGreen) * Alpha) / 255);
			BitmapBits[PixelIndex + 2] = static_cast<uint8>((static_cast<uint16>(AdjustedRed) * Alpha) / 255);
			BitmapBits[PixelIndex + 3] = Alpha;
		}
	}

	POINT DestinationPosition = { OverlayWindowX, OverlayWindowY };
	SIZE WindowSize = { RenderTargetSize.X, RenderTargetSize.Y };
	POINT SourcePosition = { 0, 0 };
	BLENDFUNCTION Blend = {};
	Blend.BlendOp = AC_SRC_OVER;
	Blend.SourceConstantAlpha = 255;
	Blend.AlphaFormat = AC_SRC_ALPHA;

	HDC ScreenDeviceContext = GetDC(nullptr);
	UpdateLayeredWindow(
		OverlayWindow,
		ScreenDeviceContext,
		&DestinationPosition,
		&WindowSize,
		MemoryDeviceContext,
		&SourcePosition,
		0,
		&Blend,
		ULW_ALPHA
	);
	ReleaseDC(nullptr, ScreenDeviceContext);
#endif
}

void AVexisPresenceGameMode::HidePrimaryGameWindow()
{
	if (GEngine == nullptr || GEngine->GameViewport == nullptr)
	{
		return;
	}

	const TSharedPtr<SWindow> GameWindow = GEngine->GameViewport->GetWindow();
	if (!GameWindow.IsValid() || !GameWindow->GetNativeWindow().IsValid())
	{
		return;
	}

#if PLATFORM_WINDOWS
	const void* NativeWindowPtr = GameWindow->GetNativeWindow()->GetOSWindowHandle();
	HWND WindowHandle = static_cast<HWND>(const_cast<void*>(NativeWindowPtr));
	if (WindowHandle == nullptr)
	{
		return;
	}

	SetWindowPos(WindowHandle, HWND_BOTTOM, -32000, -32000, 1, 1, SWP_NOACTIVATE | SWP_HIDEWINDOW);
	ShowWindow(WindowHandle, SW_HIDE);
#endif
}

bool AVexisPresenceGameMode::EnsureOverlayBitmapResources(int32 Width, int32 Height)
{
#if PLATFORM_WINDOWS
	if (Width <= 0 || Height <= 0)
	{
		return false;
	}

	if (OverlayMemoryDeviceContext != nullptr && OverlayBitmapHandle != nullptr && OverlayBitmapBits != nullptr && OverlayBitmapWidth == Width && OverlayBitmapHeight == Height)
	{
		return true;
	}

	if (OverlayBitmapHandle != nullptr)
	{
		DeleteObject(static_cast<HBITMAP>(OverlayBitmapHandle));
		OverlayBitmapHandle = nullptr;
		OverlayBitmapBits = nullptr;
	}

	if (OverlayMemoryDeviceContext != nullptr)
	{
		DeleteDC(static_cast<HDC>(OverlayMemoryDeviceContext));
		OverlayMemoryDeviceContext = nullptr;
	}

	HDC ScreenDeviceContext = GetDC(nullptr);
	HDC MemoryDeviceContext = CreateCompatibleDC(ScreenDeviceContext);
	ReleaseDC(nullptr, ScreenDeviceContext);
	if (MemoryDeviceContext == nullptr)
	{
		return false;
	}

	BITMAPINFO BitmapInfo = {};
	BitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	BitmapInfo.bmiHeader.biWidth = Width;
	BitmapInfo.bmiHeader.biHeight = -Height;
	BitmapInfo.bmiHeader.biPlanes = 1;
	BitmapInfo.bmiHeader.biBitCount = 32;
	BitmapInfo.bmiHeader.biCompression = BI_RGB;

	void* BitmapBits = nullptr;
	HBITMAP BitmapHandle = CreateDIBSection(MemoryDeviceContext, &BitmapInfo, DIB_RGB_COLORS, &BitmapBits, nullptr, 0);
	if (BitmapHandle == nullptr || BitmapBits == nullptr)
	{
		DeleteDC(MemoryDeviceContext);
		return false;
	}

	SelectObject(MemoryDeviceContext, BitmapHandle);

	OverlayMemoryDeviceContext = MemoryDeviceContext;
	OverlayBitmapHandle = BitmapHandle;
	OverlayBitmapBits = BitmapBits;
	OverlayBitmapWidth = Width;
	OverlayBitmapHeight = Height;
	return true;
#else
	return false;
#endif
}

void AVexisPresenceGameMode::SyncOverlayWindowInteraction()
{
#if PLATFORM_WINDOWS
	HWND OverlayWindow = static_cast<HWND>(OverlayWindowHandle);
	if (OverlayWindow == nullptr)
	{
		return;
	}

	const bool bShouldEnableDragMode = IsOverlayDragModifierPressed();
	if (bShouldEnableDragMode == bOverlayDragModeActive)
	{
		return;
	}

	LONG_PTR ExStyle = GetWindowLongPtrW(OverlayWindow, GWL_EXSTYLE);
	if (bShouldEnableDragMode)
	{
		ExStyle &= ~static_cast<LONG_PTR>(WS_EX_TRANSPARENT);
		bOverlayDragModeActive = true;
	}
	else
	{
		ExStyle |= static_cast<LONG_PTR>(WS_EX_TRANSPARENT);
		bOverlayDragModeActive = false;
	}

	SetWindowLongPtrW(OverlayWindow, GWL_EXSTYLE, ExStyle);
	SetWindowPos(
		OverlayWindow,
		HWND_TOPMOST,
		0,
		0,
		0,
		0,
		SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW
	);
#endif
}

void AVexisPresenceGameMode::DestroyOverlayPresenceWindow()
{
	GetWorldTimerManager().ClearTimer(OverlayRefreshTimerHandle);

#if PLATFORM_WINDOWS
	if (OverlayBitmapHandle != nullptr)
	{
		DeleteObject(static_cast<HBITMAP>(OverlayBitmapHandle));
		OverlayBitmapHandle = nullptr;
		OverlayBitmapBits = nullptr;
	}

	if (OverlayMemoryDeviceContext != nullptr)
	{
		DeleteDC(static_cast<HDC>(OverlayMemoryDeviceContext));
		OverlayMemoryDeviceContext = nullptr;
	}

	if (OverlayWindowHandle != nullptr)
	{
		DestroyWindow(static_cast<HWND>(OverlayWindowHandle));
		OverlayWindowHandle = nullptr;
	}
#endif

	OverlayBitmapWidth = 0;
	OverlayBitmapHeight = 0;
	bOverlayWindowCreated = false;
	bOverlayDragModeActive = false;
}
