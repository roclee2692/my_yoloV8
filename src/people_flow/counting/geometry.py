"""Dependency-free geometry primitives for line crossing and polygon occupancy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from people_flow.errors import CountingError

Point = tuple[float, float]


class LineSide(str, Enum):
    """Classify a point relative to the direction from line p1 to p2."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    ON_LINE = "on_line"


@dataclass(frozen=True, slots=True)
class DirectedLine:
    """Finite directed line segment with a configured enter side."""

    p1: Point
    p2: Point
    enter_side: LineSide = LineSide.POSITIVE
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.p1 == self.p2:
            raise CountingError("Counting line endpoints p1 and p2 must be different")
        if self.enter_side is LineSide.ON_LINE:
            raise CountingError("Counting line enter_side must be positive or negative")
        if self.epsilon < 0:
            raise CountingError("Counting line epsilon must be non-negative")

    @property
    def length(self) -> float:
        """Return the line-segment length in pixels."""

        return euclidean_distance(self.p1, self.p2)

    def signed_distance(self, point: Point) -> float:
        """Return normalized signed distance from the infinite directed line."""

        return cross_product(self.p1, self.p2, point) / self.length

    def side_of(self, point: Point) -> LineSide:
        """Classify a point using the configured pixel-distance tolerance."""

        distance = self.signed_distance(point)
        if distance > self.epsilon:
            return LineSide.POSITIVE
        if distance < -self.epsilon:
            return LineSide.NEGATIVE
        return LineSide.ON_LINE

    def intersects_motion(self, previous: Point, current: Point) -> bool:
        """Return whether a movement segment touches the finite counting line."""

        return segments_intersect(previous, current, self.p1, self.p2, epsilon=self.epsilon)

    def event_type_for(self, destination_side: LineSide) -> str:
        """Map a destination side to the configured enter/exit direction."""

        if destination_side is LineSide.ON_LINE:
            raise CountingError("A point on the counting line has no enter/exit direction")
        return "enter" if destination_side is self.enter_side else "exit"


@dataclass(frozen=True, slots=True)
class PolygonRegion:
    """Validated simple polygon whose boundary is considered inside."""

    name: str
    points: tuple[Point, ...]
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise CountingError("ROI name must not be blank")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "points", tuple(self.points))
        if len(self.points) < 3:
            raise CountingError("ROI polygon requires at least three points")
        if self.epsilon < 0:
            raise CountingError("ROI polygon epsilon must be non-negative")
        for point in self.points:
            if not all(math.isfinite(coordinate) for coordinate in point):
                raise CountingError("ROI polygon coordinates must be finite")
        for index, point in enumerate(self.points):
            if point == self.points[(index + 1) % len(self.points)]:
                raise CountingError("ROI polygon cannot contain consecutive duplicate points")
        self._validate_simple_polygon()
        if abs(polygon_area(self.points)) <= self.epsilon:
            raise CountingError("ROI polygon area must be greater than zero")

    def contains(self, point: Point) -> bool:
        """Return whether a point is inside or on the polygon boundary."""

        if not all(math.isfinite(coordinate) for coordinate in point):
            raise CountingError("ROI test point coordinates must be finite")
        x, y = point
        inside = False
        for index, start in enumerate(self.points):
            end = self.points[(index + 1) % len(self.points)]
            if abs(cross_product(start, end, point)) <= self.epsilon and _point_on_segment(
                point, start, end, self.epsilon
            ):
                return True
            if (start[1] > y) != (end[1] > y):
                intersection_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (
                    end[1] - start[1]
                )
                if x < intersection_x:
                    inside = not inside
        return inside

    def _validate_simple_polygon(self) -> None:
        edge_count = len(self.points)
        for first_index in range(edge_count):
            first_start = self.points[first_index]
            first_end = self.points[(first_index + 1) % edge_count]
            for second_index in range(first_index + 1, edge_count):
                if second_index == first_index:
                    continue
                if (first_index + 1) % edge_count == second_index:
                    continue
                if (second_index + 1) % edge_count == first_index:
                    continue
                second_start = self.points[second_index]
                second_end = self.points[(second_index + 1) % edge_count]
                if segments_intersect(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                    epsilon=self.epsilon,
                ):
                    raise CountingError("ROI polygon must not self-intersect")


def polygon_area(points: tuple[Point, ...]) -> float:
    """Return the signed shoelace area of an ordered polygon."""

    return 0.5 * sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(points, points[1:] + points[:1], strict=True)
    )


def cross_product(origin: Point, target: Point, point: Point) -> float:
    """Return the 2D cross product of origin->target and origin->point."""

    return (target[0] - origin[0]) * (point[1] - origin[1]) - (target[1] - origin[1]) * (
        point[0] - origin[0]
    )


def euclidean_distance(first: Point, second: Point) -> float:
    """Return pixel distance between two points."""

    return math.hypot(second[0] - first[0], second[1] - first[1])


def segments_intersect(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
    *,
    epsilon: float = 1e-6,
) -> bool:
    """Return whether two closed finite segments intersect, including endpoints."""

    first_a = cross_product(first_start, first_end, second_start)
    first_b = cross_product(first_start, first_end, second_end)
    second_a = cross_product(second_start, second_end, first_start)
    second_b = cross_product(second_start, second_end, first_end)

    if _opposite_signs(first_a, first_b, epsilon) and _opposite_signs(second_a, second_b, epsilon):
        return True
    return (
        (
            abs(first_a) <= epsilon
            and _point_on_segment(second_start, first_start, first_end, epsilon)
        )
        or (
            abs(first_b) <= epsilon
            and _point_on_segment(second_end, first_start, first_end, epsilon)
        )
        or (
            abs(second_a) <= epsilon
            and _point_on_segment(first_start, second_start, second_end, epsilon)
        )
        or (
            abs(second_b) <= epsilon
            and _point_on_segment(first_end, second_start, second_end, epsilon)
        )
    )


def _opposite_signs(first: float, second: float, epsilon: float) -> bool:
    return (first > epsilon and second < -epsilon) or (first < -epsilon and second > epsilon)


def _point_on_segment(point: Point, start: Point, end: Point, epsilon: float) -> bool:
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )
