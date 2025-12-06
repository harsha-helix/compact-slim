import os
import sys
sdk_install_path = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
python_api_path = os.path.join(sdk_install_path, "api", "python")

if python_api_path not in sys.path:
    sys.path.append(python_api_path)
    print(f"Added {python_api_path} to Python path.")

try:
    import HEDS
    from hedslib.heds_types import *
except ImportError:
    print(f"FATAL ERROR: Could not import HEDS library from {python_api_path}")
    print("Please verify the 'sdk_install_path' variable in this script.")
    sys.exit(1)
from photonic_annealing import PhotonicAnnealer, NumberPartitioningAnnealer, MaxCutAnnealer, SimulatedMaxCutAnnealer, RobustDynamicNPPAnnealer
from photonic_annealing import *
# from photonic_annealing.npp_annealer import NumberPartitioningAnnealer


print("\n" + "="*60)
print("RUNNING THREADED NPP BENCHMARK (MaxCut Style)")
print("="*60)
sizes = [50]

for size in sizes:

    NUM_SPINS = 100
    np.random.seed(42)
    weights = np.random.randint(1, 101, size=NUM_SPINS).astype(float)
    SERIAL_PORT = 'COM21' 

    annealer = None
    try:
        annealer = ThreadedDynamicNPPAnnealer(
            npp_weights=weights,
            beam_sigma_x=357,
            beam_sigma_y=357,
            serial_port=SERIAL_PORT,
            use_uint8=True 
        )
        
        # We need to monkey-patch or pass settle_time to evaluate_energy_direct
        # But evaluate_energy_direct is called inside run_annealing.
        # The class uses a default settle_time=0.005 inside the method.
        # If you need to increase it, change it in the class definition above.

        final_spins, min_energy, trace = annealer.run_annealing(
            initial_temp=1.0,
            final_temp=0.0001,
            cooling_rate=0.5, 
            steps_per_temp=30,
            Nc0=16,
            verbose=True
        )
        
        print(f"\nFinal Partition Diff: {np.dot(final_spins, weights)}")
        
        plt.figure()
        plt.plot(trace)
        plt.title("Threaded NPP Results")
        plt.show(block=True)

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
    finally:
        if annealer:
            annealer.disconnect_hardware()

    # results = {"final_spins": final_spins, "min_energy": min_energy, "trace": trace, "weights": weights}

    np.save(f"PD_benchmark_results_{time.time()}_size{size}.npy", results)
    print("Results saved to PD_benchmark_results.npy")