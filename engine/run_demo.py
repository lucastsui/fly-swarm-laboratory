"""Portable frozen viewer; starts paused unless --run is explicit."""
import argparse
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('reference', 'experimental'), default='reference')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != project:
        parser.error('Run from the repository root')
    port = 8769 if args.model == 'reference' else 8770
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            parser.error(f'Port {port} is occupied; no existing service changed')
    import uvicorn
    from .service_viewer import ServiceViewer, app_for
    common = dict(root=project/'.runtime/dopamine-haul', seed=5500000, speed=3., port=port)
    if args.model == 'reference':
        viewer = ServiceViewer(SimpleNamespace(**common,
            candidate=project/'.runtime/service-training/sequence-lr0005/candidate-200.npz',
            metadata=project/'docs/results/reference-checkpoint.json',
            evidence=project/'docs/results', flies=4,
            swarm_evidence=project/'docs/results/swarm-confirmation16-600s.json'), start_thread=False)
    else:
        from .phase_viewer import PhaseViewer
        viewer = PhaseViewer(SimpleNamespace(**common,
            phase=project/'.runtime/dashboard/latest-phase.json',
            playback='paced', inference_backend='cached-csr'), start_thread=False)
    viewer.running = args.run
    if hasattr(viewer, 'publish_state'):
        viewer.publish_state()
    threading.Thread(target=viewer.loop, daemon=True).start()
    if hasattr(viewer, 'watch'):
        threading.Thread(target=viewer.watch, daemon=True).start()
    print(f'Frozen {args.model} viewer on 127.0.0.1:{port}; running={args.run}; learning=False', flush=True)
    uvicorn.run(app_for(viewer), host='127.0.0.1', port=port, access_log=False)

if __name__ == '__main__':
    main()
