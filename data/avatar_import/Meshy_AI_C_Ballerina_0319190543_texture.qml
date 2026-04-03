import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    property url textureData: "maps/textureData.jpg"
    property url textureData7: "maps/textureData7.jpg"
    property url textureData9: "maps/textureData9.jpg"
    Texture {
        id: _0_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData
    }
    Texture {
        id: _1_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData7
    }
    Texture {
        id: _2_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData9
    }
    PrincipledMaterial {
        id: principledMaterial
        baseColorMap: _0_texture
        metalnessMap: _1_texture
        roughnessMap: _1_texture
        metalness: 1
        roughness: 1
        normalMap: _2_texture
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
    }

    // Nodes:
    Model {
        id: nodes_0_
        objectName: "nodes[0]"
        source: "meshes/meshes_0__mesh.mesh"
        materials: [
            principledMaterial
        ]
    }

    // Animations:
}
