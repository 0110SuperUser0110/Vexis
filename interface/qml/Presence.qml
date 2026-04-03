import QtQuick
import QtQuick3D

Item {
    id: root
    width: 680
    height: 760

    property string stateName: "idle"
    property var thoughtLines: []
    property color accentColor: stateName === "speaking" ? "#f0abfc"
                               : stateName === "listening" ? "#d8b4fe"
                               : stateName === "thinking" ? "#c084fc"
                               : stateName === "dancing" ? "#f472b6"
                               : stateName === "error" ? "#fb7185"
                               : "#9333ea"
    property color accentSoft: stateName === "speaking" ? "#fdf4ff"
                              : stateName === "listening" ? "#f5edff"
                              : stateName === "thinking" ? "#f3e8ff"
                              : stateName === "dancing" ? "#fce7f3"
                              : stateName === "error" ? "#ffe4e6"
                              : "#ede9fe"
    property color eyeColor: stateName === "error" ? "#ffe4e6" : "#faf5ff"

    View3D {
        anchors.fill: parent
        camera: camera

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.VeryHigh
        }

        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0, 0.14, 3.55)
            clipNear: 0.1
            clipFar: 40
        }

        DirectionalLight {
            brightness: 3.4
            eulerRotation.x: -28
            eulerRotation.y: -20
            color: root.accentSoft
            ambientColor: "#12081d"
        }

        PointLight {
            position: Qt.vector3d(1.1, 0.8, 2.2)
            brightness: stateName === "speaking" ? 22.0 : 16.0
            color: root.accentColor
            quadraticFade: 0.58
        }

        PointLight {
            position: Qt.vector3d(-1.15, 0.25, 1.7)
            brightness: stateName === "thinking" ? 13.0 : 8.5
            color: "#6d28d9"
            quadraticFade: 0.86
        }

        PointLight {
            position: Qt.vector3d(0.0, -0.95, 1.3)
            brightness: 4.4
            color: root.accentSoft
            quadraticFade: 1.05
        }

        Node {
            id: rig
            position: Qt.vector3d(0, -0.18, 0)
            scale: Qt.vector3d(2.05, 2.05, 2.05)

            SequentialAnimation on y {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: stateName === "dancing" ? -0.28 : -0.22
                    to: stateName === "dancing" ? -0.04 : -0.12
                    duration: stateName === "dancing" ? 520 : (stateName === "thinking" ? 1800 : 2800)
                    easing.type: Easing.InOutSine
                }
                NumberAnimation {
                    from: stateName === "dancing" ? -0.04 : -0.12
                    to: stateName === "dancing" ? -0.28 : -0.22
                    duration: stateName === "dancing" ? 520 : (stateName === "thinking" ? 1800 : 2800)
                    easing.type: Easing.InOutSine
                }
            }

            SequentialAnimation on eulerRotation.y {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: stateName === "dancing" ? -16 : -5
                    to: stateName === "dancing" ? 16 : 5
                    duration: stateName === "dancing" ? 420 : (stateName === "speaking" ? 1400 : 3200)
                    easing.type: Easing.InOutSine
                }
                NumberAnimation {
                    from: stateName === "dancing" ? 16 : 5
                    to: stateName === "dancing" ? -16 : -5
                    duration: stateName === "dancing" ? 420 : (stateName === "speaking" ? 1400 : 3200)
                    easing.type: Easing.InOutSine
                }
            }

            SequentialAnimation on eulerRotation.x {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: stateName === "dancing" ? -7.0 : -1.6
                    to: stateName === "dancing" ? 7.8 : 2.2
                    duration: stateName === "dancing" ? 680 : 2400
                    easing.type: Easing.InOutSine
                }
                NumberAnimation {
                    from: stateName === "dancing" ? 7.8 : 2.2
                    to: stateName === "dancing" ? -7.0 : -1.6
                    duration: stateName === "dancing" ? 680 : 2400
                    easing.type: Easing.InOutSine
                }
            }

            SequentialAnimation on eulerRotation.z {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: stateName === "dancing" ? -10 : -1.5
                    to: stateName === "dancing" ? 10 : 1.5
                    duration: stateName === "dancing" ? 360 : 3000
                    easing.type: Easing.InOutSine
                }
                NumberAnimation {
                    from: stateName === "dancing" ? 10 : 1.5
                    to: stateName === "dancing" ? -10 : -1.5
                    duration: stateName === "dancing" ? 360 : 3000
                    easing.type: Easing.InOutSine
                }
            }

            PrincipledMaterial {
                id: headCoreMaterial
                baseColor: Qt.rgba(root.accentColor.r, root.accentColor.g, root.accentColor.b, 0.26)
                metalness: 0.08
                roughness: 0.16
                specularAmount: 0.98
                indexOfRefraction: 1.18
                opacity: 0.34
                alphaMode: PrincipledMaterial.Blend
                cullMode: PrincipledMaterial.NoCulling
                emissiveFactor: stateName === "speaking"
                                ? Qt.vector3d(0.82, 0.42, 1.0)
                                : stateName === "thinking"
                                  ? Qt.vector3d(0.62, 0.28, 0.84)
                                  : stateName === "dancing"
                                    ? Qt.vector3d(0.94, 0.38, 0.88)
                                    : stateName === "error"
                                      ? Qt.vector3d(1.0, 0.22, 0.26)
                                      : Qt.vector3d(0.52, 0.20, 0.70)
            }

            PrincipledMaterial {
                id: headShellMaterial
                baseColor: Qt.rgba(root.accentSoft.r, root.accentSoft.g, root.accentSoft.b, 0.12)
                metalness: 0.0
                roughness: 0.04
                specularAmount: 0.94
                indexOfRefraction: 1.06
                opacity: 0.10
                alphaMode: PrincipledMaterial.Blend
                cullMode: PrincipledMaterial.NoCulling
                emissiveFactor: stateName === "speaking"
                                ? Qt.vector3d(1.0, 0.78, 1.0)
                                : stateName === "thinking"
                                  ? Qt.vector3d(0.92, 0.60, 1.0)
                                  : stateName === "dancing"
                                    ? Qt.vector3d(1.0, 0.78, 0.96)
                                    : stateName === "error"
                                      ? Qt.vector3d(1.0, 0.40, 0.48)
                                      : Qt.vector3d(0.80, 0.48, 1.0)
            }

            PrincipledMaterial {
                id: eyeMaterial
                baseColor: Qt.rgba(root.eyeColor.r, root.eyeColor.g, root.eyeColor.b, 0.40)
                metalness: 0.0
                roughness: 0.02
                specularAmount: 1.0
                opacity: 0.72
                alphaMode: PrincipledMaterial.Blend
                cullMode: PrincipledMaterial.NoCulling
                emissiveFactor: stateName === "speaking"
                                ? Qt.vector3d(1.0, 0.94, 1.0)
                                : stateName === "thinking"
                                  ? Qt.vector3d(0.90, 0.86, 1.0)
                                  : stateName === "dancing"
                                    ? Qt.vector3d(1.0, 0.88, 0.98)
                                    : stateName === "error"
                                      ? Qt.vector3d(1.0, 0.66, 0.70)
                                      : Qt.vector3d(0.86, 0.82, 1.0)
            }

            Node {
                id: headAsset

                Model {
                    source: "../../data/avatar_head_import/meshes/female_head_head_mesh.mesh"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    materials: [ headCoreMaterial ]
                }

                Model {
                    source: "../../data/avatar_head_import/meshes/female_head_head_mesh.mesh"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(1.016, 1.016, 1.016)
                    materials: [ headShellMaterial ]
                }

                Model {
                    source: "../../data/avatar_head_import/meshes/female_head_eyeLeft_mesh.mesh"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    materials: [ eyeMaterial ]
                }

                Model {
                    source: "../../data/avatar_head_import/meshes/female_head_eyeRight_mesh.mesh"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    materials: [ eyeMaterial ]
                }
            }
        }
    }
}
