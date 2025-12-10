#!/usr/bin/env python3
"""
Unified launcher for all three neural workspace simulations
Choose between different optimized versions and run them together
"""

import sys
import os

def run_optimized_with_heatmaps():
    """Run the final optimized version with beautiful heatmaps"""
    print("Starting Final Animation with Heatmaps...")
    print("   Features: Real-time heatmaps, optimized performance, beautiful visualization")
    
    try:
        from final_animation_with_heatmaps import animate_neural_workspace_with_heatmaps
        animate_neural_workspace_with_heatmaps()
    except Exception as e:
        print(f"Error running heatmap version: {e}")
        print("Falling back to basic 3-model animation...")
        run_basic_three_model()

def run_all_three_optimized():
    """Run the performance-optimized version"""
    print("Starting All Three Optimized Simulations...")
    print("   Features: Maximum performance, caching, optimized metrics")
    
    try:
        from lab_all_three_optimized import animate_all_three_optimized
        animate_all_three_optimized()
    except Exception as e:
        print(f"Error running optimized version: {e}")
        print("Falling back to basic 3-model animation...")
        run_basic_three_model()

def run_basic_three_model():
    """Run the original comprehensive version"""
    print("Starting Original 3-Model Animation...")
    print("   Features: Complete implementation, all original features")
    
    try:
        from lab import animate_workspace_heatmap_forever
        animate_workspace_heatmap_forever()
    except Exception as e:
        print(f"Error running basic version: {e}")
        print("Trying to run individual components...")
        run_individual_components()

def run_individual_components():
    """Run individual simulation components for testing"""
    print("Running Individual Component Tests...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Test basic functionality
        print("Testing basic numpy and matplotlib...")
        print(f"NumPy version: {np.__version__}")
        print(f"Matplotlib version: {plt.matplotlib.__version__}")
        
        # Simple test plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        ax.set_title('Basic Test Plot - Libraries Working!')
        ax.set_xlabel('x')
        ax.set_ylabel('sin(x)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Basic plotting works! Libraries are functional.")
        
    except Exception as e:
        print(f"Error with basic components: {e}")
        print("Please check your Python environment and dependencies.")

def run_performance_comparison():
    """Run performance comparison of all three models"""
    print("Running Performance Comparison...")
    
    try:
        from lab_all_three_optimized import compare_all_three_performance
        compare_all_three_performance()
    except Exception as e:
        print(f"Error running performance comparison: {e}")

def show_menu():
    """Show interactive menu for choosing simulation"""
    print("\n" + "="*60)
    print("NEURAL WORKSPACE: 3-MODEL SIMULATION LAUNCHER")
    print("="*60)
    print("Choose your simulation:")
    print()
    print("1. Final Animation with Heatmaps (RECOMMENDED)")
    print("   -> Beautiful real-time heatmaps")
    print("   -> Optimized performance") 
    print("   -> Best visualization")
    print()
    print("2. All Three Optimized (Performance Mode)")
    print("   -> Maximum speed and efficiency")
    print("   -> Advanced caching system")
    print("   -> Performance metrics")
    print()
    print("3. Original 3-Model (Complete Features)")
    print("   -> Original comprehensive implementation")
    print("   -> All original features")
    print("   -> Full functionality")
    print()
    print("4. Performance Comparison")
    print("   -> Compare speed of all models")
    print("   -> Benchmark testing")
    print()
    print("5. Test Individual Components")
    print("   -> Test basic functionality")
    print("   -> Debugging help")
    print()
    print("6. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                run_optimized_with_heatmaps()
                break
            elif choice == '2':
                run_all_three_optimized()
                break
            elif choice == '3':
                run_basic_three_model()
                break
            elif choice == '4':
                run_performance_comparison()
                break
            elif choice == '5':
                run_individual_components()
                break
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

def main():
    """Main launcher function"""
    print("Neural Workspace Simulation Launcher")
    print("Ready to run all three simulations together!")
    print()
    
    # Check if command line argument provided
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['1', 'heatmaps', 'visualization', 'final']:
            run_optimized_with_heatmaps()
        elif arg in ['2', 'optimized', 'performance', 'fast']:
            run_all_three_optimized()
        elif arg in ['3', 'original', 'basic', 'classic']:
            run_basic_three_model()
        elif arg in ['4', 'performance', 'benchmark', 'compare']:
            run_performance_comparison()
        elif arg in ['5', 'test', 'debug', 'components']:
            run_individual_components()
        else:
            print(f"Unknown argument: {arg}")
            print("Available options: 1, 2, 3, 4, 5, or no argument for interactive mode")
    else:
        # Interactive mode
        show_menu()

if __name__ == "__main__":
    main()