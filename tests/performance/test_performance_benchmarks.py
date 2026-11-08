"""
Performance tests for Infinite Maze.

These tests verify performance characteristics, benchmarks,
and resource usage under various conditions.
"""

import pytest
import time
import gc
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.core.clock import Clock
from infinite_maze.utils.config import GameConfig
from tests.fixtures.pygame_mocks import full_pygame_mocks
from tests.fixtures.test_helpers import PerformanceMonitor


class TestFrameRatePerformance:
    """Test frame rate and timing performance."""
    
    @pytest.mark.performance
    def test_target_60_fps_performance(self):
        """Test maintaining 60 FPS target."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generateMaze(game, 15, 20)
            
            monitor = PerformanceMonitor()
            target_fps = 60.0
            frame_time_budget = 1000.0 / target_fps  # ~16.67ms per frame
            
            frame_times = []
            
            # Measure frame times for 2 seconds
            for frame in range(120):  # 2 seconds at 60 FPS
                frame_start = time.perf_counter()
                
                # Simulate game loop
                player.moveX(1 if frame % 4 == 0 else 0)
                if frame % 4 == 0:
                    game.incrementScore()
                
                game.updateScreen(player, lines)
                game.getClock().update()
                
                frame_end = time.perf_counter()
                frame_time_ms = (frame_end - frame_start) * 1000
                frame_times.append(frame_time_ms)
            
            # Analyze frame performance
            avg_frame_time = sum(frame_times) / len(frame_times)
            max_frame_time = max(frame_times)
            min_frame_time = min(frame_times)
            
            # Performance assertions
            assert avg_frame_time < frame_time_budget, f"Average frame time {avg_frame_time:.2f}ms exceeds budget {frame_time_budget:.2f}ms"
            assert max_frame_time < frame_time_budget * 2, f"Max frame time {max_frame_time:.2f}ms too high"
            assert min_frame_time > 0, "Invalid frame time measurement"
            
            # Calculate achieved FPS
            achieved_fps = 1000.0 / avg_frame_time
            assert achieved_fps > 45.0, f"FPS too low: {achieved_fps:.1f}"
    
    @pytest.mark.performance
    def test_frame_consistency(self):
        """Test frame time consistency (low variance)."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 10, 15)
            
            frame_times = []
            
            # Measure consistent workload
            for frame in range(180):  # 3 seconds
                frame_start = time.perf_counter()
                
                # Consistent workload
                player.moveX(1)
                game.incrementScore()
                game.updateScreen(player, lines)
                game.getClock().update()
                
                frame_end = time.perf_counter()
                frame_times.append((frame_end - frame_start) * 1000)
            
            # Calculate variance
            avg_time = sum(frame_times) / len(frame_times)
            variance = sum((t - avg_time) ** 2 for t in frame_times) / len(frame_times)
            std_dev = variance ** 0.5
            
            # Frame times should be consistent
            coefficient_of_variation = std_dev / avg_time
            assert coefficient_of_variation < 0.5, f"Frame times too inconsistent: CV={coefficient_of_variation:.3f}"
    
    @pytest.mark.performance
    def test_high_pace_fps_stability(self):
        """Test FPS stability at high game pace."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            lines = Line.generateMaze(game, 20, 25)
            
            # Set high pace
            game.setPace(8)
            
            frame_times = []
            
            for frame in range(300):  # 5 seconds at 60 FPS
                frame_start = time.perf_counter()
                
                # High-pace workload
                pace = game.getPace()
                for line in lines:
                    line.setXStart(line.getXStart() - pace)
                    line.setXEnd(line.getXEnd() - pace)
                
                player.moveX(pace // 2)
                game.incrementScore()
                game.updateScreen(player, lines)
                game.getClock().update()
                
                frame_end = time.perf_counter()
                frame_times.append((frame_end - frame_start) * 1000)
            
            # Even at high pace, should maintain reasonable FPS
            avg_frame_time = sum(frame_times) / len(frame_times)
            achieved_fps = 1000.0 / avg_frame_time
            
            assert achieved_fps > 30.0, f"High-pace FPS too low: {achieved_fps:.1f}"


class TestMemoryPerformance:
    """Test memory usage and leak detection."""
    
    @pytest.mark.performance
    def test_memory_baseline(self):
        """Test baseline memory usage."""
        # Measure baseline
        gc.collect()
        initial_memory = PerformanceMonitor().get_memory_usage_mb()['current']
        
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 10, 15)
            
            # Measure after initialization
            gc.collect()
            post_init_memory = PerformanceMonitor().get_memory_usage_mb()['current']
            
            # Memory increase should be reasonable
            memory_increase = post_init_memory - initial_memory
            assert memory_increase < 50, f"Memory increase too high: {memory_increase:.1f}MB"
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks during gameplay."""
        with full_pygame_mocks():
            monitor = PerformanceMonitor()
            
            # Baseline measurement
            gc.collect()
            baseline_memory = monitor.get_memory_usage_mb()['current']
            
            # Run multiple game sessions
            for session in range(5):
                game = Game(headless=True)
                player = Player(100 + session * 20, 200, headless=True)
                lines = Line.generateMaze(game, 15, 20)
                
                # Play session
                for frame in range(300):  # 5 second session
                    player.moveX(1 if frame % 3 == 0 else 0)
                    if frame % 3 == 0:
                        game.incrementScore()
                    
                    game.updateScreen(player, lines)
                    game.getClock().update()
                
                # Cleanup
                game.cleanup()
                del game, player, lines
                gc.collect()
                
                # Sample memory
                monitor.sample_memory()
            
            # Final memory check
            final_memory = monitor.get_memory_usage_mb()['current']
            memory_growth = final_memory - baseline_memory
            
            # Should not have significant memory growth
            assert memory_growth < 20, f"Possible memory leak detected: {memory_growth:.1f}MB growth"
    
    @pytest.mark.performance
    def test_large_maze_memory_usage(self):
        """Test memory usage with large mazes."""
        with full_pygame_mocks():
            monitor = PerformanceMonitor()
            
            gc.collect()
            baseline_memory = monitor.get_memory_usage_mb()['current']
            
            # Create progressively larger mazes
            maze_sizes = [(10, 15), (20, 25), (30, 35), (40, 45), (50, 55)]
            
            for width, height in maze_sizes:
                game = Game(headless=True)
                lines = Line.generateMaze(game, width, height)
                
                monitor.sample_memory()
                
                # Memory should scale reasonably
                current_memory = monitor.get_memory_usage_mb()['current']
                memory_usage = current_memory - baseline_memory
                
                # Rule of thumb: each line should use < 1KB on average
                expected_max_memory = (width * height * 0.001) + 10  # 10MB baseline
                assert memory_usage < expected_max_memory, f"Memory usage too high for {width}x{height}: {memory_usage:.1f}MB"
                
                del game, lines
                gc.collect()
    
    @pytest.mark.performance  
    def test_player_creation_memory_efficiency(self):
        """Test memory efficiency of player creation."""
        with full_pygame_mocks():
            monitor = PerformanceMonitor()
            
            gc.collect()
            baseline_memory = monitor.get_memory_usage_mb()['current']
            
            # Create many players
            players = []
            for i in range(100):
                player = Player(100 + i, 200 + i, headless=True)
                players.append(player)
                
                if i % 20 == 0:
                    monitor.sample_memory()
            
            # Check memory usage
            peak_memory = monitor.get_memory_usage_mb()['peak']
            memory_per_player = (peak_memory - baseline_memory) / 100
            
            # Each player should use minimal memory
            assert memory_per_player < 0.1, f"Memory per player too high: {memory_per_player:.3f}MB"
            
            # Cleanup
            players.clear()
            gc.collect()


class TestComputationalPerformance:
    """Test computational performance of algorithms."""
    
    @pytest.mark.performance
    def test_collision_detection_performance(self):
        """Test collision detection algorithm performance."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 30, 40)  # Large maze
            
            # Benchmark collision detection
            start_time = time.perf_counter()
            
            collision_count = 0
            test_iterations = 10000
            
            for iteration in range(test_iterations):
                # Test movement in each direction
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_x = player.getX() + dx * player.getSpeed()
                    test_y = player.getY() + dy * player.getSpeed()
                    
                    # Check collision with all lines
                    for line in lines:
                        if line.getIsHorizontal():
                            if (test_y <= line.getYStart() <= test_y + player.getHeight() and
                                test_x < line.getXEnd() and test_x + player.getWidth() > line.getXStart()):
                                collision_count += 1
                        else:
                            if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                                test_y < line.getYEnd() and test_y + player.getHeight() > line.getYStart()):
                                collision_count += 1
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Performance requirements
            checks_per_second = (test_iterations * 4 * len(lines)) / duration
            assert checks_per_second > 1000000, f"Collision detection too slow: {checks_per_second:.0f} checks/sec"
            assert duration < 5.0, f"Collision detection took too long: {duration:.2f}s"
    
    @pytest.mark.performance
    def test_maze_generation_performance(self):
        """Test maze generation algorithm performance."""
        with full_pygame_mocks():
            game = Game(headless=True)
            
            # Benchmark maze generation
            generation_times = []
            maze_sizes = [(10, 15), (20, 25), (30, 35), (40, 45)]
            
            for width, height in maze_sizes:
                start_time = time.perf_counter()
                
                # Generate multiple mazes of this size
                for _ in range(10):
                    lines = Line.generateMaze(game, width, height)
                    assert len(lines) > 0
                
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / 10
                generation_times.append((width * height, avg_time))
            
            # Verify reasonable scaling
            for size, gen_time in generation_times:
                time_per_cell = gen_time / size
                assert time_per_cell < 0.001, f"Maze generation too slow: {time_per_cell:.4f}s per cell"
    
    @pytest.mark.performance
    def test_score_calculation_performance(self):
        """Test score calculation performance."""
        with full_pygame_mocks():
            game = Game(headless=True)
            
            # Benchmark score operations
            start_time = time.perf_counter()
            
            operations = 1000000
            for i in range(operations):
                if i % 2 == 0:
                    game.incrementScore()
                else:
                    game.decrementScore()
                
                # Occasionally check score
                if i % 1000 == 0:
                    score = game.getScore()
                    assert score >= 0
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            ops_per_second = operations / duration
            assert ops_per_second > 100000, f"Score operations too slow: {ops_per_second:.0f} ops/sec"
    
    @pytest.mark.performance
    def test_clock_update_performance(self):
        """Test clock update performance."""
        clock = Clock()
        
        start_time = time.perf_counter()
        
        updates = 100000
        for _ in range(updates):
            clock.update()
            
            # Occasionally access clock values
            if _ % 1000 == 0:
                millis = clock.getMillis()
                seconds = clock.getSeconds()
                assert millis >= 0
                assert seconds >= 0
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        updates_per_second = updates / duration
        assert updates_per_second > 50000, f"Clock updates too slow: {updates_per_second:.0f} updates/sec"


class TestConcurrencyPerformance:
    """Test performance under concurrent conditions."""
    
    @pytest.mark.performance
    def test_thread_safety_performance(self):
        """Test performance with multiple threads accessing game components."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            
            results = []
            errors = []
            
            def worker_thread(thread_id):
                try:
                    for i in range(1000):
                        # Simulate concurrent access
                        score = game.getScore()
                        pos = player.getPosition()
                        
                        # Safe operations only
                        if thread_id == 0:  # Only one thread modifies
                            if i % 10 == 0:
                                game.incrementScore()
                        
                        results.append((thread_id, i, score, pos))
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # Run concurrent threads
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(4)]
                
                for future in futures:
                    future.result()  # Wait for completion
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Verify results
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert duration < 5.0, f"Concurrent access too slow: {duration:.2f}s"
            assert len(results) == 4000, "Missing results from concurrent access"
    
    @pytest.mark.performance
    def test_rapid_state_changes_performance(self):
        """Test performance with rapid state changes."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            
            start_time = time.perf_counter()
            
            # Rapid state changes
            for i in range(10000):
                # Rapid movement
                direction = i % 4
                if direction == 0:
                    player.moveX(1)
                elif direction == 1:
                    player.moveX(-1)
                elif direction == 2:
                    player.moveY(1)
                else:
                    player.moveY(-1)
                
                # Rapid score changes
                if i % 2 == 0:
                    game.incrementScore()
                else:
                    game.decrementScore()
                
                # Rapid pause/unpause
                if i % 100 == 0:
                    game.changePaused(player)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            changes_per_second = 30000 / duration  # 3 changes per iteration
            assert changes_per_second > 10000, f"State changes too slow: {changes_per_second:.0f} changes/sec"


class TestScalabilityPerformance:
    """Test performance scalability."""
    
    @pytest.mark.performance
    def test_maze_size_scalability(self):
        """Test performance scaling with maze size."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            performance_data = []
            
            # Test different maze sizes
            sizes = [(5, 8), (10, 15), (20, 25), (30, 35), (40, 45)]
            
            for width, height in sizes:
                lines = Line.generateMaze(game, width, height)
                
                # Measure update performance
                start_time = time.perf_counter()
                
                for frame in range(100):
                    game.updateScreen(player, lines)
                    player.moveX(1)
                    game.incrementScore()
                
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                maze_size = width * height
                performance_data.append((maze_size, duration))
                
                # Performance should not degrade too badly
                fps = 100 / duration
                assert fps > 30, f"FPS too low for {width}x{height} maze: {fps:.1f}"
            
            # Check scaling characteristics
            for i in range(1, len(performance_data)):
                prev_size, prev_time = performance_data[i-1]
                curr_size, curr_time = performance_data[i]
                
                size_ratio = curr_size / prev_size
                time_ratio = curr_time / prev_time
                
                # Time should not increase faster than O(n^2)
                assert time_ratio < size_ratio ** 2, f"Performance scaling too poor: {time_ratio:.2f}x time for {size_ratio:.2f}x size"
    
    @pytest.mark.performance
    def test_game_duration_scalability(self):
        """Test performance over extended game duration."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generateMaze(game, 15, 20)
            
            monitor = PerformanceMonitor()
            
            # Simulate 10 minutes of gameplay
            frame_count = 36000  # 10 minutes at 60 FPS
            performance_samples = []
            
            for frame in range(frame_count):
                frame_start = time.perf_counter()
                
                # Standard gameplay
                if frame % 4 == 0:
                    player.moveX(1)
                    game.incrementScore()
                
                game.updateScreen(player, lines)
                game.getClock().update()
                
                frame_end = time.perf_counter()
                frame_time = (frame_end - frame_start) * 1000
                
                # Sample every 10 seconds
                if frame % 600 == 0:
                    performance_samples.append(frame_time)
                    monitor.sample_memory()
            
            # Verify stable performance over time
            early_samples = performance_samples[:3]
            late_samples = performance_samples[-3:]
            
            early_avg = sum(early_samples) / len(early_samples)
            late_avg = sum(late_samples) / len(late_samples)
            
            performance_degradation = late_avg / early_avg
            assert performance_degradation < 1.5, f"Performance degraded too much: {performance_degradation:.2f}x"
            
            # Memory should remain stable
            memory_stats = monitor.get_memory_usage_mb()
            memory_growth = memory_stats['peak'] - memory_stats['samples'][0] if memory_stats['samples'] else 0
            assert memory_growth < 50, f"Memory grew too much: {memory_growth:.1f}MB"


class TestResourceUtilizationPerformance:
    """Test resource utilization efficiency."""
    
    @pytest.mark.performance
    def test_cpu_utilization_efficiency(self):
        """Test CPU utilization efficiency."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 20, 25)
            
            # Measure active vs idle CPU usage
            start_time = time.perf_counter()
            active_start = time.process_time()
            
            # Run game for specific duration
            for frame in range(600):  # 10 seconds at 60 FPS
                player.moveX(1 if frame % 3 == 0 else 0)
                if frame % 3 == 0:
                    game.incrementScore()
                
                game.updateScreen(player, lines)
                game.getClock().update()
                
                # Simulate frame timing
                time.sleep(0.001)  # Small sleep to simulate real timing
            
            end_time = time.perf_counter()
            active_end = time.process_time()
            
            wall_time = end_time - start_time
            cpu_time = active_end - active_start
            
            # CPU efficiency should be reasonable
            cpu_utilization = cpu_time / wall_time
            assert cpu_utilization < 0.8, f"CPU utilization too high: {cpu_utilization:.2f}"
            assert cpu_utilization > 0.1, f"CPU utilization suspiciously low: {cpu_utilization:.2f}"
    
    @pytest.mark.performance
    def test_object_allocation_efficiency(self):
        """Test object allocation efficiency."""
        with full_pygame_mocks():
            # Track object creation
            allocation_count = 0
            
            # Monitor allocations during gameplay
            start_objects = len(gc.get_objects())
            
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Short gameplay session
            for frame in range(300):
                player.moveX(1 if frame % 2 == 0 else 0)
                game.updateScreen(player, [])
                game.getClock().update()
            
            end_objects = len(gc.get_objects())
            object_growth = end_objects - start_objects
            
            # Object growth should be minimal for steady-state gameplay
            assert object_growth < 1000, f"Too many objects allocated: {object_growth}"
    
    @pytest.mark.performance
    def test_config_access_performance(self):
        """Test configuration access performance."""
        config = GameConfig()
        
        start_time = time.perf_counter()
        
        # Rapid config access
        for _ in range(10000):
            colors = [
                config.getRedColor(),
                config.getGreenColor(),
                config.getBlueColor(),
                config.getWhiteColor(),
                config.getBlackColor()
            ]
            
            paths = [
                config.getPlayerImage(),
                config.getPlayerPausedImage(),
                config.getIcon()
            ]
            
            # Verify all values returned
            assert all(len(color) == 3 for color in colors)
            assert all(isinstance(path, str) for path in paths)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        accesses_per_second = 80000 / duration  # 8 accesses per iteration
        assert accesses_per_second > 50000, f"Config access too slow: {accesses_per_second:.0f} accesses/sec"


class TestBenchmarkSuite:
    """Comprehensive benchmark suite."""
    
    @pytest.mark.performance
    def test_overall_performance_benchmark(self):
        """Overall performance benchmark test."""
        with full_pygame_mocks():
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Comprehensive benchmark
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 25, 30)
            
            benchmark_results = {}
            
            # Test 1: Basic gameplay
            start_time = time.perf_counter()
            for frame in range(300):
                player.moveX(1 if frame % 2 == 0 else 0)
                if frame % 2 == 0:
                    game.incrementScore()
                game.updateScreen(player, lines)
                game.getClock().update()
            basic_time = time.perf_counter() - start_time
            benchmark_results['basic_gameplay'] = 300 / basic_time  # FPS
            
            # Test 2: High-pace gameplay
            game.setPace(5)
            start_time = time.perf_counter()
            for frame in range(300):
                for line in lines:
                    line.setXStart(line.getXStart() - 5)
                    line.setXEnd(line.getXEnd() - 5)
                player.moveX(2)
                game.incrementScore()
                game.updateScreen(player, lines)
                game.getClock().update()
            high_pace_time = time.perf_counter() - start_time
            benchmark_results['high_pace_gameplay'] = 300 / high_pace_time  # FPS
            
            # Test 3: Collision-heavy scenario
            dense_lines = Line.generateMaze(game, 40, 50)
            start_time = time.perf_counter()
            for frame in range(200):
                # Test collision detection
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_x = player.getX() + dx
                    test_y = player.getY() + dy
                    
                    for line in dense_lines[:20]:  # Test subset for speed
                        # Collision check
                        if line.getIsHorizontal():
                            if (test_y <= line.getYStart() <= test_y + player.getHeight() and
                                test_x < line.getXEnd() and test_x + player.getWidth() > line.getXStart()):
                                pass
                        else:
                            if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                                test_y < line.getYEnd() and test_y + player.getHeight() > line.getYStart()):
                                pass
                
                game.updateScreen(player, dense_lines)
                game.getClock().update()
            collision_time = time.perf_counter() - start_time
            benchmark_results['collision_heavy'] = 200 / collision_time  # FPS
            
            monitor.stop()
            
            # Benchmark requirements
            assert benchmark_results['basic_gameplay'] > 60, f"Basic gameplay too slow: {benchmark_results['basic_gameplay']:.1f} FPS"
            assert benchmark_results['high_pace_gameplay'] > 45, f"High-pace gameplay too slow: {benchmark_results['high_pace_gameplay']:.1f} FPS"
            assert benchmark_results['collision_heavy'] > 30, f"Collision-heavy gameplay too slow: {benchmark_results['collision_heavy']:.1f} FPS"
            
            # Memory efficiency
            memory_stats = monitor.get_memory_usage_mb()
            assert memory_stats['peak'] < 200, f"Memory usage too high: {memory_stats['peak']:.1f}MB"
            
            print(f"\n=== Performance Benchmark Results ===")
            print(f"Basic Gameplay: {benchmark_results['basic_gameplay']:.1f} FPS")
            print(f"High-Pace Gameplay: {benchmark_results['high_pace_gameplay']:.1f} FPS")
            print(f"Collision-Heavy: {benchmark_results['collision_heavy']:.1f} FPS")
            print(f"Peak Memory: {memory_stats['peak']:.1f}MB")
            print(f"Total Duration: {monitor.get_duration():.2f}s")


class TestSimulationThroughput:
    """The rules run without sleeping or wall-clock dependence."""

    @pytest.mark.performance
    def test_ten_thousand_steps_run_without_delay(self):
        from infinite_maze.core.simulation import Simulation

        sim = Simulation(seed=1, max_steps=None)
        intents = [(1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

        with patch("pygame.time.delay") as delay:
            start = time.perf_counter()
            for step in range(10000):
                sim.step(intents[step % len(intents)])
            duration = time.perf_counter() - start

        delay.assert_not_called()
        # Real play paces frames at ~16ms, so 10k frames would take ~160s.
        assert duration < 60.0
