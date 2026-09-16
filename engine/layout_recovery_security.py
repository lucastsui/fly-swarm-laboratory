"""Create experiment-scoped cluster credentials; never prints secret contents."""
import argparse
import os
import secrets
import subprocess
from pathlib import Path


def main(directory):
    if directory.exists():
        raise FileExistsError('Never replace existing cluster credentials')
    directory.mkdir(mode=0o700, parents=True)
    token = directory/'token'
    token.write_text(secrets.token_urlsafe(48))
    os.chmod(token, 0o600)
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(directory/'key.pem'), '-out', str(directory/'cert.pem'),
                    '-days', '14', '-subj', '/CN=fly-layout-recovery-cluster'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chmod(directory/'key.pem', 0o600)
    print('Experiment TLS certificate and token created; private cluster link only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
