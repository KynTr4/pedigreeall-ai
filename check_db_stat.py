import paramiko
import sys

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        stdin, stdout, stderr = ssh.exec_command("stat /opt/at_yaris_tahmini/pedigreeall_progress.db")
        print("STAT OUTPUT:")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
