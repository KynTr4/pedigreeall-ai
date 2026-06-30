import paramiko

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        stdin, stdout, stderr = ssh.exec_command("df -h")
        print("DF OUTPUT:")
        print(stdout.read().decode('utf-8'))
        
        stdin, stdout, stderr = ssh.exec_command("du -h --max-depth=1 /var/backups/at_yaris_tahmini || true")
        print("BACKUPS SIZE:")
        print(stdout.read().decode('utf-8'))
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
