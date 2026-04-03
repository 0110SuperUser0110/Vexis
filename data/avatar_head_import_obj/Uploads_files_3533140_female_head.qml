import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: initialShadingGroup_material
        objectName: "initialShadingGroup"
        baseColor: "#ff999999"
        indexOfRefraction: 1
    }

    // Nodes:
    Node {
        id: uploads_files_3533140_female_head_obj
        objectName: "uploads_files_3533140_female_head.obj"
        Node {
            id: default_
            objectName: "default"
        }
        Model {
            id: female_head_eyeLeft
            objectName: "female_head:eyeLeft"
            source: "meshes/female_head_eyeLeft_mesh.mesh"
            materials: [
                initialShadingGroup_material
            ]
        }
        Node {
            id: default_6
            objectName: "default"
        }
        Model {
            id: female_head_eyeRight
            objectName: "female_head:eyeRight"
            source: "meshes/female_head_eyeRight_mesh.mesh"
            materials: [
                initialShadingGroup_material
            ]
        }
        Node {
            id: default_9
            objectName: "default"
        }
        Model {
            id: female_head_head
            objectName: "female_head:head"
            source: "meshes/female_head_head_mesh.mesh"
            materials: [
                initialShadingGroup_material
            ]
        }
    }

    // Animations:
}
