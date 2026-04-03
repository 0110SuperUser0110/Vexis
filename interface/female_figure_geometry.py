from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import QVector3D
from PySide6.QtQuick3D import QQuick3DGeometry


@dataclass(frozen=True)
class BodySection:
    y: float
    radius_x: float
    radius_front: float
    radius_back: float
    z_offset: float = 0.0


class FemaleFigureGeometry(QQuick3DGeometry):
    """Procedurally generated translucent female figure mesh for the VEXIS presence window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_geometry()

    def _build_geometry(self) -> None:
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []

        def append_mesh(mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> None:
            if mesh_vertices.size == 0 or mesh_faces.size == 0:
                return
            offset = sum(part.shape[0] for part in vertices)
            vertices.append(mesh_vertices.astype(np.float32, copy=False))
            faces.append((mesh_faces + offset).astype(np.uint32, copy=False))

        torso_sections = [
            BodySection(0.98, 0.075, 0.060, 0.055),
            BodySection(0.88, 0.110, 0.080, 0.070),
            BodySection(0.76, 0.270, 0.125, 0.110),
            BodySection(0.64, 0.255, 0.235, 0.125, 0.012),
            BodySection(0.52, 0.245, 0.260, 0.120, 0.016),
            BodySection(0.40, 0.205, 0.175, 0.110, 0.010),
            BodySection(0.18, 0.175, 0.145, 0.100, 0.006),
            BodySection(-0.04, 0.145, 0.112, 0.095, 0.002),
            BodySection(-0.24, 0.205, 0.135, 0.135, -0.006),
            BodySection(-0.42, 0.275, 0.170, 0.195, -0.018),
            BodySection(-0.60, 0.245, 0.145, 0.165, -0.014),
            BodySection(-0.78, 0.115, 0.105, 0.120, -0.010),
        ]
        torso_v, torso_f = self._create_loft(torso_sections, segments=40, cap_top=True, cap_bottom=True)
        append_mesh(torso_v, torso_f)

        head_v, head_f = self._create_ellipsoid(
            center=np.array([0.0, 1.22, 0.02], dtype=np.float32),
            radii=np.array([0.155, 0.220, 0.165], dtype=np.float32),
            lat_steps=22,
            lon_steps=36,
        )
        append_mesh(head_v, head_f)

        hair_v, hair_f = self._create_ellipsoid(
            center=np.array([0.0, 1.24, -0.008], dtype=np.float32),
            radii=np.array([0.185, 0.245, 0.188], dtype=np.float32),
            lat_steps=22,
            lon_steps=36,
            radial_perturbation=lambda lat, lon: 1.0 + 0.06 * math.sin(lat * 2.3) * math.sin(lon * 3.1) + 0.03 * math.cos(lon * 8.0),
        )
        append_mesh(hair_v, hair_f)

        for center in (
            np.array([-0.090, 0.57, 0.19], dtype=np.float32),
            np.array([0.090, 0.57, 0.19], dtype=np.float32),
        ):
            bust_v, bust_f = self._create_ellipsoid(
                center=center,
                radii=np.array([0.105, 0.125, 0.115], dtype=np.float32),
                lat_steps=16,
                lon_steps=28,
            )
            append_mesh(bust_v, bust_f)

        pelvis_v, pelvis_f = self._create_ellipsoid(
            center=np.array([0.0, -0.47, -0.02], dtype=np.float32),
            radii=np.array([0.24, 0.15, 0.19], dtype=np.float32),
            lat_steps=14,
            lon_steps=28,
        )
        append_mesh(pelvis_v, pelvis_f)

        left_upper_arm = self._create_tube(
            start=np.array([-0.27, 0.72, 0.04], dtype=np.float32),
            end=np.array([-0.07, 0.40, 0.17], dtype=np.float32),
            radius_start=0.060,
            radius_end=0.050,
            segments=24,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*left_upper_arm)
        left_forearm = self._create_tube(
            start=np.array([-0.07, 0.40, 0.17], dtype=np.float32),
            end=np.array([0.24, 0.55, 0.26], dtype=np.float32),
            radius_start=0.046,
            radius_end=0.036,
            segments=24,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*left_forearm)
        left_hand = self._create_ellipsoid(
            center=np.array([0.34, 0.62, 0.21], dtype=np.float32),
            radii=np.array([0.050, 0.090, 0.030], dtype=np.float32),
            lat_steps=10,
            lon_steps=16,
        )
        append_mesh(*left_hand)

        right_upper_arm = self._create_tube(
            start=np.array([0.27, 0.72, 0.04], dtype=np.float32),
            end=np.array([0.07, 0.36, 0.15], dtype=np.float32),
            radius_start=0.060,
            radius_end=0.050,
            segments=24,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*right_upper_arm)
        right_forearm = self._create_tube(
            start=np.array([0.07, 0.36, 0.15], dtype=np.float32),
            end=np.array([-0.23, 0.50, 0.24], dtype=np.float32),
            radius_start=0.046,
            radius_end=0.036,
            segments=24,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*right_forearm)
        right_hand = self._create_ellipsoid(
            center=np.array([-0.33, 0.57, 0.20], dtype=np.float32),
            radii=np.array([0.050, 0.090, 0.030], dtype=np.float32),
            lat_steps=10,
            lon_steps=16,
        )
        append_mesh(*right_hand)

        left_thigh = self._create_tube(
            start=np.array([-0.110, -0.69, 0.00], dtype=np.float32),
            end=np.array([-0.125, -1.22, 0.03], dtype=np.float32),
            radius_start=0.100,
            radius_end=0.058,
            segments=28,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*left_thigh)
        left_calf = self._create_tube(
            start=np.array([-0.125, -1.22, 0.03], dtype=np.float32),
            end=np.array([-0.108, -1.78, 0.00], dtype=np.float32),
            radius_start=0.054,
            radius_end=0.030,
            segments=28,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*left_calf)
        left_foot = self._create_ellipsoid(
            center=np.array([-0.102, -1.90, 0.075], dtype=np.float32),
            radii=np.array([0.055, 0.030, 0.110], dtype=np.float32),
            lat_steps=10,
            lon_steps=18,
        )
        append_mesh(*left_foot)

        right_thigh = self._create_tube(
            start=np.array([0.110, -0.69, 0.00], dtype=np.float32),
            end=np.array([0.125, -1.22, 0.03], dtype=np.float32),
            radius_start=0.100,
            radius_end=0.058,
            segments=28,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*right_thigh)
        right_calf = self._create_tube(
            start=np.array([0.125, -1.22, 0.03], dtype=np.float32),
            end=np.array([0.108, -1.78, 0.00], dtype=np.float32),
            radius_start=0.054,
            radius_end=0.030,
            segments=28,
            cap_start=True,
            cap_end=True,
        )
        append_mesh(*right_calf)
        right_foot = self._create_ellipsoid(
            center=np.array([0.102, -1.90, 0.075], dtype=np.float32),
            radii=np.array([0.055, 0.030, 0.110], dtype=np.float32),
            lat_steps=10,
            lon_steps=18,
        )
        append_mesh(*right_foot)

        joint_centers = [
            (-0.07, 0.40, 0.17),
            (0.07, 0.36, 0.15),
            (-0.125, -1.22, 0.03),
            (0.125, -1.22, 0.03),
        ]
        for joint in joint_centers:
            joint_v, joint_f = self._create_ellipsoid(
                center=np.array(joint, dtype=np.float32),
                radii=np.array([0.045, 0.045, 0.045], dtype=np.float32),
                lat_steps=12,
                lon_steps=18,
            )
            append_mesh(joint_v, joint_f)

        vertex_array = np.concatenate(vertices, axis=0)
        face_array = np.concatenate(faces, axis=0)
        normals = self._compute_normals(vertex_array, face_array)
        interleaved = np.concatenate([vertex_array, normals], axis=1).astype(np.float32, copy=False)
        index_data = face_array.astype(np.uint32, copy=False).ravel()

        self.clear()
        self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Triangles)
        self.setStride(24)
        self.setVertexData(interleaved.tobytes())
        self.setIndexData(index_data.tobytes())
        self.addAttribute(QQuick3DGeometry.Attribute.Semantic.PositionSemantic, 0, QQuick3DGeometry.Attribute.ComponentType.F32Type)
        self.addAttribute(QQuick3DGeometry.Attribute.Semantic.NormalSemantic, 12, QQuick3DGeometry.Attribute.ComponentType.F32Type)
        self.addAttribute(QQuick3DGeometry.Attribute.Semantic.IndexSemantic, 0, QQuick3DGeometry.Attribute.ComponentType.U32Type)

        mins = vertex_array.min(axis=0)
        maxs = vertex_array.max(axis=0)
        self.setBounds(
            QVector3D(float(mins[0]), float(mins[1]), float(mins[2])),
            QVector3D(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        )

    def _create_loft(
        self,
        sections: list[BodySection],
        segments: int,
        cap_top: bool,
        cap_bottom: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        rings: list[np.ndarray] = []
        for section in sections:
            ring = []
            for index in range(segments):
                angle = (2.0 * math.pi * index) / segments
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                front_back = section.radius_front if sin_a >= 0.0 else section.radius_back
                ring.append(
                    [
                        section.radius_x * cos_a,
                        section.y,
                        section.z_offset + front_back * sin_a,
                    ]
                )
            rings.append(np.array(ring, dtype=np.float32))

        vertices = np.vstack(rings)
        faces: list[tuple[int, int, int]] = []
        ring_count = len(rings)

        for ring_index in range(ring_count - 1):
            base = ring_index * segments
            next_base = (ring_index + 1) * segments
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                a = base + segment
                b = base + next_segment
                c = next_base + segment
                d = next_base + next_segment
                faces.append((a, c, b))
                faces.append((b, c, d))

        if cap_top:
            top_center = rings[0].mean(axis=0)
            top_index = len(vertices)
            vertices = np.vstack([vertices, top_center])
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                faces.append((top_index, next_segment, segment))

        if cap_bottom:
            bottom_center = rings[-1].mean(axis=0)
            bottom_index = len(vertices)
            vertices = np.vstack([vertices, bottom_center])
            base = (ring_count - 1) * segments
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                faces.append((bottom_index, base + segment, base + next_segment))

        return vertices.astype(np.float32), np.array(faces, dtype=np.uint32)

    def _create_ellipsoid(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        lat_steps: int,
        lon_steps: int,
        radial_perturbation=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices: list[list[float]] = []
        faces: list[tuple[int, int, int]] = []

        for lat in range(lat_steps + 1):
            phi = math.pi * lat / lat_steps
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            for lon in range(lon_steps):
                theta = 2.0 * math.pi * lon / lon_steps
                scale = radial_perturbation(phi, theta) if radial_perturbation else 1.0
                x = center[0] + radii[0] * scale * sin_phi * math.cos(theta)
                y = center[1] + radii[1] * scale * cos_phi
                z = center[2] + radii[2] * scale * sin_phi * math.sin(theta)
                vertices.append([x, y, z])

        for lat in range(lat_steps):
            for lon in range(lon_steps):
                next_lon = (lon + 1) % lon_steps
                a = lat * lon_steps + lon
                b = lat * lon_steps + next_lon
                c = (lat + 1) * lon_steps + lon
                d = (lat + 1) * lon_steps + next_lon
                if lat != 0:
                    faces.append((a, c, b))
                if lat != lat_steps - 1:
                    faces.append((b, c, d))

        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)

    def _create_tube(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius_start: float,
        radius_end: float,
        segments: int,
        cap_start: bool,
        cap_end: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        axis = end - start
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1e-6:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)

        direction = axis / axis_length
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(direction, helper))) > 0.92:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tangent = np.cross(direction, helper)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(direction, tangent)
        bitangent /= np.linalg.norm(bitangent)

        start_ring = []
        end_ring = []
        for index in range(segments):
            angle = (2.0 * math.pi * index) / segments
            ring_direction = math.cos(angle) * tangent + math.sin(angle) * bitangent
            start_ring.append(start + ring_direction * radius_start)
            end_ring.append(end + ring_direction * radius_end)

        vertices = np.vstack([np.array(start_ring, dtype=np.float32), np.array(end_ring, dtype=np.float32)])
        faces: list[tuple[int, int, int]] = []

        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = segment
            b = next_segment
            c = segments + segment
            d = segments + next_segment
            faces.append((a, c, b))
            faces.append((b, c, d))

        if cap_start:
            center_index = len(vertices)
            vertices = np.vstack([vertices, start.astype(np.float32)])
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                faces.append((center_index, next_segment, segment))

        if cap_end:
            center_index = len(vertices)
            vertices = np.vstack([vertices, end.astype(np.float32)])
            base = segments
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                faces.append((center_index, base + segment, base + next_segment))

        return vertices.astype(np.float32), np.array(faces, dtype=np.uint32)

    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(vertices, dtype=np.float32)
        for face in faces:
            p0 = vertices[face[0]]
            p1 = vertices[face[1]]
            p2 = vertices[face[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            length = np.linalg.norm(normal)
            if length > 1e-8:
                normal /= length
                normals[face[0]] += normal
                normals[face[1]] += normal
                normals[face[2]] += normal

        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths < 1e-8] = 1.0
        normals /= lengths[:, None]
        return normals.astype(np.float32)
