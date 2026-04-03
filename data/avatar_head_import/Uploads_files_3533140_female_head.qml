import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: lambert1_material
        objectName: "lambert1"
        baseColor: "#ff666666"
    }

    // Nodes:
    Node {
        id: rootNode
        objectName: "RootNode"
        Model {
            id: female_head_eyeLeft
            objectName: "female_head:eyeLeft"
            rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
            source: "meshes/female_head_eyeLeft_mesh.mesh"
            materials: [
                lambert1_material
            ]
        }
        Model {
            id: female_head_eyeRight
            objectName: "female_head:eyeRight"
            rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
            source: "meshes/female_head_eyeRight_mesh.mesh"
            materials: [
                lambert1_material
            ]
        }
        Model {
            id: female_head_head
            objectName: "female_head:head"
            rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
            source: "meshes/female_head_head_mesh.mesh"
            materials: [
                lambert1_material
            ]
        }
    }

    // Animations:
}
