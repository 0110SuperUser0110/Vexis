import QtQuick
import QtQuick3D

Item {
    id: root
    width: 520
    height: 620

    property string stateName: "idle"
    property var thoughtLines: ["VEXIS online", "Awaiting input"]

    function randomBinary(length) {
        var s = ""
        for (var i = 0; i < length; i++) {
            s += Math.random() < 0.5 ? "0" : "1"
            if ((i + 1) % 6 === 0 && i < length - 1)
                s += "\n"
        }
        return s
    }

    View3D {
        anchors.fill: parent
        camera: cam

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
        }

        PerspectiveCamera {
            id: cam
            position: Qt.vector3d(0, 0, 300)
            clipNear: 1
            clipFar: 2000
        }

        // Main hard key light from top-right, aimed leftward
        DirectionalLight {
            brightness: 4.6
            eulerRotation.x: -70
            eulerRotation.y: -60
            color: "#fff8ff"
            ambientColor: "#221733"

            castsShadow: true
            shadowFactor: 100
            shadowBias: 6
            shadowMapQuality: Light.ShadowMapQualityVeryHigh
        }

        // Gentle fill from the upper-right so faces still read cleanly
        PointLight {
            position: Qt.vector3d(180, 150, 220)
            brightness: 0.55
            color: "#caa3ff"
            constantFade: 1.0
            linearFade: 0.0
            quadraticFade: 0.18
        }

        // Very dim opposite-side fill so the dark side is not fully crushed
        PointLight {
            position: Qt.vector3d(-120, -90, 170)
            brightness: 0.15
            color: "#241a38"
            constantFade: 1.0
            linearFade: 0.0
            quadraticFade: 0.35
        }

        Node {
            id: pivot
            position: Qt.vector3d(0, 0, 0)

            PropertyAnimation on eulerRotation.y {
                from: 0
                to: 360
                duration: stateName === "thinking" ? 18000
                         : stateName === "speaking" ? 16500
                         : stateName === "listening" ? 17500
                         : 19000
                loops: Animation.Infinite
                running: true
            }

            SequentialAnimation on y {
                loops: Animation.Infinite
                running: true

                NumberAnimation {
                    from: stateName === "thinking" ? -6 : -5
                    to: stateName === "thinking" ? 6 : 5
                    duration: stateName === "thinking" ? 2400 : 2800
                    easing.type: Easing.InOutSine
                }

                NumberAnimation {
                    from: stateName === "thinking" ? 6 : 5
                    to: stateName === "thinking" ? -6 : -5
                    duration: stateName === "thinking" ? 2400 : 2800
                    easing.type: Easing.InOutSine
                }
            }

            Model {
                id: vexCube
                source: "#Cube"
                position: Qt.vector3d(0, 0, 0)
                scale: Qt.vector3d(
                    stateName === "thinking" ? 1.53 : 1.46,
                    stateName === "thinking" ? 1.53 : 1.46,
                    stateName === "thinking" ? 1.53 : 1.46
                )
                eulerRotation: Qt.vector3d(22, 28, 8)

                Behavior on scale {
                    Vector3dAnimation { duration: 320 }
                }

                materials: [
                    PrincipledMaterial {
                        baseColor: stateName === "thinking" ? "#22052d"
                                  : stateName === "speaking" ? "#55215f"
                                  : stateName === "listening" ? "#20335e"
                                  : "#3b1f59"

                        emissiveFactor: stateName === "thinking"
                                        ? Qt.vector3d(0.04, 0.01, 0.06)
                                        : stateName === "speaking"
                                          ? Qt.vector3d(0.06, 0.02, 0.08)
                                          : stateName === "listening"
                                            ? Qt.vector3d(0.02, 0.03, 0.06)
                                            : Qt.vector3d(0.02, 0.01, 0.04)

                        metalness: 0.70
                        roughness: 0.42
                        specularAmount: 0.65
                        indexOfRefraction: 1.45
                    }
                ]
            }
        }
    }

    Item {
        id: binaryLayer
        visible: stateName === "thinking"
        width: 270
        height: 270
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        clip: true
        opacity: 0.95

        Repeater {
            model: 11

            Text {
                id: stream
                property int idx: index
                property int bitLength: 36 + Math.floor(Math.random() * 18)
                property real laneX: 10 + (idx * 24)
                property real startY: -220 - (idx * 18)
                property int fallDuration: 5200 + Math.floor(Math.random() * 2400)
                property int fadeDuration: 1500 + Math.floor(Math.random() * 800)

                text: root.randomBinary(bitLength)
                color: idx % 3 === 0 ? "#ff8df3"
                      : idx % 3 === 1 ? "#ea6cff"
                      : "#c94cff"
                opacity: 0.22 + (Math.random() * 0.32)
                font.family: "Consolas"
                font.pixelSize: 20
                font.bold: true
                lineHeight: 0.82
                x: laneX
                y: startY
                rotation: 0

                style: Text.Outline
                styleColor: "#2a002d"

                function refreshBits() {
                    text = root.randomBinary(bitLength)
                }

                Timer {
                    interval: 1100 + (stream.idx * 90)
                    repeat: true
                    running: stateName === "thinking"
                    onTriggered: stream.refreshBits()
                }

                SequentialAnimation on y {
                    loops: Animation.Infinite
                    running: stateName === "thinking"

                    NumberAnimation {
                        from: stream.startY
                        to: binaryLayer.height + 40
                        duration: stream.fallDuration
                        easing.type: Easing.Linear
                    }

                    ScriptAction {
                        script: {
                            stream.refreshBits()
                            stream.y = stream.startY
                        }
                    }
                }

                SequentialAnimation on opacity {
                    loops: Animation.Infinite
                    running: stateName === "thinking"

                    NumberAnimation { from: 0.22; to: 0.72; duration: stream.fadeDuration }
                    NumberAnimation { from: 0.72; to: 0.26; duration: stream.fadeDuration + 350 }
                }
            }
        }
    }

    Text {
        text: "VEXIS"
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 24
        color: "#f2ecff"
        font.pixelSize: 22
        font.bold: true
    }

    Text {
        text: stateName === "thinking"
              ? "state: thinking | binary stream active"
              : "state: " + stateName
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 24
        color: "#cabaff"
        font.pixelSize: 14
    }
}