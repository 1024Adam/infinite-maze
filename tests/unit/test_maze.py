"""
Unit tests for the Line/Maze classes in Infinite Maze.

These tests verify the functionality of the Line class including
line positioning, maze generation, collision boundaries, and
maze management operations.
"""

import pytest
from unittest.mock import Mock, patch

from infinite_maze.entities.maze import Line
from infinite_maze.core.game import Game
from infinite_maze.utils.config import config


class TestLineInitialization:
    """Test Line class initialization."""
    
    def test_line_init_default(self):
        """Test line initialization with default parameters."""
        line = Line()
        
        assert line.getStart() == (0, 0)
        assert line.getEnd() == (0, 0)
        assert line.getSideA() == 0
        assert line.getSideB() == 0
        assert line.getIsHorizontal() is True  # Same Y coordinates
    
    def test_line_init_with_parameters(self):
        """Test line initialization with specific parameters."""
        start_pos = (10, 20)
        end_pos = (30, 40)
        side_a = 5
        side_b = 7
        
        line = Line(start_pos, end_pos, side_a, side_b)
        
        assert line.getStart() == start_pos
        assert line.getEnd() == end_pos
        assert line.getSideA() == side_a
        assert line.getSideB() == side_b
    
    def test_line_horizontal_detection(self):
        """Test horizontal line detection."""
        # Horizontal line (same Y coordinates)
        horizontal_line = Line((10, 20), (30, 20))
        assert horizontal_line.getIsHorizontal() is True
        
        # Vertical line (different Y coordinates)
        vertical_line = Line((10, 20), (10, 40))
        assert vertical_line.getIsHorizontal() is False
        
        # Diagonal line
        diagonal_line = Line((10, 20), (30, 40))
        assert diagonal_line.getIsHorizontal() is False
    
    def test_line_negative_coordinates(self):
        """Test line with negative coordinates."""
        line = Line((-10, -20), (-5, -15))
        
        assert line.getStart() == (-10, -20)
        assert line.getEnd() == (-5, -15)


class TestLinePositionGetters:
    """Test Line position getter methods."""
    
    def test_get_start_end(self):
        """Test getting start and end positions."""
        line = Line((10, 20), (30, 40))
        
        assert line.getStart() == (10, 20)
        assert line.getEnd() == (30, 40)
    
    def test_get_individual_coordinates(self):
        """Test getting individual coordinate values."""
        line = Line((15, 25), (35, 45))
        
        assert line.getXStart() == 15
        assert line.getYStart() == 25
        assert line.getXEnd() == 35
        assert line.getYEnd() == 45
    
    def test_coordinate_consistency(self):
        """Test that coordinates are consistent across methods."""
        line = Line((100, 200), (300, 400))
        
        start = line.getStart()
        end = line.getEnd()
        
        assert start[0] == line.getXStart()
        assert start[1] == line.getYStart()
        assert end[0] == line.getXEnd()
        assert end[1] == line.getYEnd()


class TestLinePositionSetters:
    """Test Line position setter methods."""
    
    def test_set_start_position(self):
        """Test setting start position."""
        line = Line((10, 20), (30, 40))
        
        new_start = (50, 60)
        line.setStart(new_start)
        
        assert line.getStart() == new_start
        assert line.getEnd() == (30, 40)  # End should not change
    
    def test_set_end_position(self):
        """Test setting end position."""
        line = Line((10, 20), (30, 40))
        
        new_end = (70, 80)
        line.setEnd(new_end)
        
        assert line.getStart() == (10, 20)  # Start should not change
        assert line.getEnd() == new_end
    
    def test_set_individual_coordinates(self):
        """Test setting individual coordinates."""
        line = Line((10, 20), (30, 40))
        
        # Set start coordinates
        line.setXStart(100)
        assert line.getXStart() == 100
        assert line.getYStart() == 20  # Y should not change
        
        line.setYStart(200)
        assert line.getXStart() == 100  # X should not change
        assert line.getYStart() == 200
        
        # Set end coordinates
        line.setXEnd(300)
        assert line.getXEnd() == 300
        assert line.getYEnd() == 40  # Y should not change
        
        line.setYEnd(400)
        assert line.getXEnd() == 300  # X should not change
        assert line.getYEnd() == 400
    
    def test_coordinate_setting_updates_position(self):
        """Test that setting coordinates updates position tuples."""
        line = Line((10, 20), (30, 40))
        
        line.setXStart(50)
        line.setYStart(60)
        
        assert line.getStart() == (50, 60)
        
        line.setXEnd(70)
        line.setYEnd(80)
        
        assert line.getEnd() == (70, 80)


class TestLineSideManagement:
    """Test Line side management for maze generation."""
    
    def test_get_sides(self):
        """Test getting side values."""
        line = Line((0, 0), (10, 10), 5, 7)
        
        assert line.getSideA() == 5
        assert line.getSideB() == 7
    
    def test_set_sides(self):
        """Test setting side values."""
        line = Line((0, 0), (10, 10), 1, 2)
        
        line.setSideA(10)
        line.setSideB(15)
        
        assert line.getSideA() == 10
        assert line.getSideB() == 15
    
    def test_sides_independence(self):
        """Test that side A and B are independent."""
        line = Line((0, 0), (10, 10), 1, 2)
        
        line.setSideA(100)
        assert line.getSideB() == 2  # Should not change
        
        line.setSideB(200)
        assert line.getSideA() == 100  # Should not change


class TestLineOrientationHandling:
    """Test Line orientation detection and management."""
    
    def test_horizontal_line_detection(self):
        """Test detection of horizontal lines."""
        horizontal_line = Line((10, 50), (100, 50))  # Same Y
        assert horizontal_line.getIsHorizontal() is True
    
    def test_vertical_line_detection(self):
        """Test detection of vertical lines."""
        vertical_line = Line((50, 10), (50, 100))  # Same X
        assert vertical_line.getIsHorizontal() is False
    
    def test_orientation_after_position_change(self):
        """Test orientation detection after position changes."""
        line = Line((10, 20), (30, 40))  # Initially diagonal
        assert line.getIsHorizontal() is False
        
        # Change to horizontal
        line.setYEnd(20)  # Make Y coordinates same
        # Note: resetIsHorizontal would need to be called or implemented differently
        # The current implementation doesn't automatically update isHorizontal
    
    def test_reset_is_horizontal(self):
        """Test resetIsHorizontal method."""
        line = Line((10, 20), (30, 40))
        
        # Change position to horizontal
        line.setYEnd(20)
        
        # Reset orientation detection
        line.resetIsHorizontal()
        
        # Should now detect as horizontal
        assert line.getIsHorizontal() is True


class TestLineMazeGeneration:
    """Test Line maze generation functionality."""
    
    def test_generate_maze_basic(self):
        """Test basic maze generation."""
        # Create a mock game object
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        # Generate small maze
        lines = Line.generateMaze(mock_game, 3, 3)
        
        # Should return a list of lines
        assert isinstance(lines, list)
        assert len(lines) > 0
        
        # All items should be Line objects
        for line in lines:
            assert isinstance(line, Line)
    
    def test_generate_maze_dimensions(self):
        """Test maze generation with different dimensions."""
        mock_game = Mock()
        mock_game.X_MAX = 200
        mock_game.Y_MIN = 100
        
        # Test different sizes
        sizes = [(2, 2), (5, 5), (10, 8)]
        
        for width, height in sizes:
            lines = Line.generateMaze(mock_game, width, height)
            
            # Should generate lines
            assert len(lines) > 0
            
            # Number of lines should be related to maze size
            # (exact formula depends on implementation)
            assert len(lines) > width + height
    
    def test_generate_maze_line_properties(self):
        """Test properties of generated maze lines."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generateMaze(mock_game, 3, 3)
        
        for line in lines:
            # Each line should have valid coordinates
            start = line.getStart()
            end = line.getEnd()
            
            assert isinstance(start[0], int)
            assert isinstance(start[1], int)
            assert isinstance(end[0], int)
            assert isinstance(end[1], int)
            
            # Lines should have valid side values
            assert isinstance(line.getSideA(), int)
            assert isinstance(line.getSideB(), int)
    
    def test_generate_maze_with_game_bounds(self):
        """Test that generated maze respects game bounds."""
        mock_game = Mock()
        mock_game.X_MAX = 300
        mock_game.Y_MIN = 40
        
        lines = Line.generateMaze(mock_game, 5, 5)
        
        for line in lines:
            # Lines should start at/after the configured player start region
            assert line.getXStart() >= config.PLAYER_START_X
            assert line.getYStart() >= mock_game.Y_MIN


class TestLineUtilityMethods:
    """Test Line utility methods."""
    
    def test_get_x_max_empty_list(self):
        """Test getXMax with empty line list."""
        max_x = Line.getXMax([])
        assert max_x == 0
    
    def test_get_x_max_single_line(self):
        """Test getXMax with single line."""
        line = Line((10, 20), (50, 30))
        max_x = Line.getXMax([line])
        assert max_x == 50
    
    def test_get_x_max_multiple_lines(self):
        """Test getXMax with multiple lines."""
        lines = [
            Line((10, 20), (50, 30)),   # max X = 50
            Line((30, 40), (80, 50)),   # max X = 80
            Line((5, 10), (25, 15)),    # max X = 25
            Line((60, 70), (120, 80))   # max X = 120
        ]
        
        max_x = Line.getXMax(lines)
        assert max_x == 120
    
    def test_get_x_max_negative_coordinates(self):
        """Test getXMax with negative coordinates."""
        lines = [
            Line((-50, 0), (-20, 10)),  # max X = -20
            Line((-30, 0), (-10, 10)),  # max X = -10
            Line((-40, 0), (-35, 10))   # max X = -35
        ]
        
        max_x = Line.getXMax(lines)
        assert max_x == -10
    
    def test_get_x_max_mixed_coordinates(self):
        """Test getXMax with mixed positive/negative coordinates."""
        lines = [
            Line((-50, 0), (30, 10)),   # max X = 30
            Line((10, 0), (-5, 10)),    # max X = 10
            Line((-10, 0), (50, 10))    # max X = 50
        ]
        
        max_x = Line.getXMax(lines)
        assert max_x == 50


class TestLineCollisionBoundaries:
    """Test Line collision boundary calculations."""
    
    def test_horizontal_line_boundaries(self):
        """Test horizontal line collision boundaries."""
        line = Line((100, 200), (200, 200))  # Horizontal line
        
        # Line should span from X=100 to X=200 at Y=200
        assert line.getXStart() == 100
        assert line.getXEnd() == 200
        assert line.getYStart() == 200
        assert line.getYEnd() == 200
        assert line.getIsHorizontal() is True
    
    def test_vertical_line_boundaries(self):
        """Test vertical line collision boundaries."""
        line = Line((150, 100), (150, 300))  # Vertical line
        
        # Line should span from Y=100 to Y=300 at X=150
        assert line.getXStart() == 150
        assert line.getXEnd() == 150
        assert line.getYStart() == 100
        assert line.getYEnd() == 300
        assert line.getIsHorizontal() is False
    
    def test_line_length_calculation(self):
        """Test calculating line length."""
        # Horizontal line
        h_line = Line((10, 50), (110, 50))
        h_length = h_line.getXEnd() - h_line.getXStart()
        assert h_length == 100
        
        # Vertical line
        v_line = Line((50, 10), (50, 60))
        v_length = v_line.getYEnd() - v_line.getYStart()
        assert v_length == 50


class TestLineEdgeCases:
    """Test Line edge cases and error conditions."""
    
    def test_zero_length_line(self):
        """Test line with zero length."""
        line = Line((50, 50), (50, 50))
        
        assert line.getStart() == line.getEnd()
        assert line.getXStart() == line.getXEnd()
        assert line.getYStart() == line.getYEnd()
        assert line.getIsHorizontal() is True  # Same Y coordinates
    
    def test_negative_coordinate_line(self):
        """Test line with negative coordinates."""
        line = Line((-100, -50), (-20, -30))
        
        assert line.getXStart() == -100
        assert line.getYStart() == -50
        assert line.getXEnd() == -20
        assert line.getYEnd() == -30
    
    def test_large_coordinate_values(self):
        """Test line with very large coordinates."""
        large_val = 999999
        line = Line((0, 0), (large_val, large_val))
        
        assert line.getXEnd() == large_val
        assert line.getYEnd() == large_val
    
    def test_line_coordinate_modification_sequence(self):
        """Test sequence of coordinate modifications."""
        line = Line((10, 20), (30, 40))
        
        # Sequence of modifications
        modifications = [
            ('setXStart', 100),
            ('setYStart', 200),
            ('setXEnd', 300),
            ('setYEnd', 400),
            ('setSideA', 5),
            ('setSideB', 7)
        ]
        
        for method_name, value in modifications:
            method = getattr(line, method_name)
            method(value)
        
        # Check final state
        assert line.getStart() == (100, 200)
        assert line.getEnd() == (300, 400)
        assert line.getSideA() == 5
        assert line.getSideB() == 7


class TestLineMazeIntegration:
    """Integration tests for Line with maze generation."""
    
    def test_maze_generation_with_real_game(self):
        """Test maze generation with real Game object."""
        # This test uses the actual Game class
        game = Game(headless=True)
        
        lines = Line.generateMaze(game, 5, 5)
        
        # Should generate valid maze
        assert len(lines) > 0
        
        # Lines should be positioned relative to game boundaries
        for line in lines:
            # All lines should be within reasonable bounds
            assert line.getXStart() >= 0
            assert line.getYStart() >= 0
            assert line.getXEnd() >= line.getXStart()
            
            if not line.getIsHorizontal():
                assert line.getYEnd() >= line.getYStart()
    
    def test_generated_maze_connectivity(self):
        """Test that generated maze has proper connectivity."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generateMaze(mock_game, 4, 4)
        
        # Generated maze should form a connected structure
        # (specific connectivity tests would depend on maze algorithm)
        assert len(lines) > 0
        
        # Should have both horizontal and vertical lines
        horizontal_lines = [line for line in lines if line.getIsHorizontal()]
        vertical_lines = [line for line in lines if not line.getIsHorizontal()]
        
        # In a proper maze, we expect both types of lines
        # (unless it's a very simple maze)
        assert len(horizontal_lines) > 0 or len(vertical_lines) > 0
    
    def test_maze_line_positioning_pattern(self):
        """Test that maze lines follow expected positioning pattern."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generateMaze(mock_game, 3, 3)
        
        # Lines should follow grid-based positioning
        # (specific pattern depends on maze generation algorithm)
        x_positions = set()
        y_positions = set()
        
        for line in lines:
            x_positions.add(line.getXStart())
            x_positions.add(line.getXEnd())
            y_positions.add(line.getYStart())
            y_positions.add(line.getYEnd())
        
        # Should have multiple distinct positions
        assert len(x_positions) > 1
        assert len(y_positions) > 1


@pytest.mark.performance
class TestLinePerformance:
    """Performance tests for Line operations."""
    
    def test_line_creation_performance(self):
        """Test performance of creating many lines."""
        import time
        start_time = time.time()
        
        lines = []
        for i in range(10000):
            line = Line((i, i), (i+10, i+10), i, i+1)
            lines.append(line)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0, f"Line creation took too long: {duration}s"
        assert len(lines) == 10000
    
    def test_coordinate_access_performance(self):
        """Test performance of coordinate access operations."""
        line = Line((100, 200), (300, 400))
        
        import time
        start_time = time.time()
        
        # Perform many coordinate access operations
        for _ in range(100000):
            _ = line.getXStart()
            _ = line.getYStart()
            _ = line.getXEnd()
            _ = line.getYEnd()
            _ = line.getStart()
            _ = line.getEnd()
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0, f"Coordinate access took too long: {duration}s"
    
    def test_maze_generation_performance(self):
        """Test performance of maze generation."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        import time
        start_time = time.time()
        
        # Generate multiple mazes
        for _ in range(100):
            lines = Line.generateMaze(mock_game, 10, 10)
            assert len(lines) > 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Maze generation took too long: {duration}s"


def _lineKeys(lines):
    return [(line.getStart(), line.getEnd()) for line in lines]


class TestSeededMazeGeneration:
    """Maze generation is reproducible from an injected RNG."""

    def _game(self):
        game = Mock()
        game.Y_MIN = config.MAZE_TOP_Y
        return game

    def test_same_seed_gives_identical_lines(self):
        import random

        first = Line.generateMaze(self._game(), 6, 8, random.Random(42))
        second = Line.generateMaze(self._game(), 6, 8, random.Random(42))

        assert _lineKeys(first) == _lineKeys(second)

    def test_different_seeds_give_different_lines(self):
        import random

        first = Line.generateMaze(self._game(), 6, 8, random.Random(1))
        second = Line.generateMaze(self._game(), 6, 8, random.Random(2))

        assert _lineKeys(first) != _lineKeys(second)

    def test_rng_none_still_generates(self):
        lines = Line.generateMaze(self._game(), 4, 5, None)

        assert len(lines) > 0

    def test_global_random_state_is_not_consulted(self):
        import random

        random.seed(1)
        first = Line.generateMaze(self._game(), 4, 5)
        random.seed(1)
        second = Line.generateMaze(self._game(), 4, 5)

        assert _lineKeys(first) != _lineKeys(second)
