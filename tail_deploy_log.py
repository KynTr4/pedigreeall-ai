import paramiko

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        stdin, stdout, stderr = ssh.exec_command("tail -n 30 /tmp/deploy.log || echo 'no log yet'")
        print("DEPLOY LOG TAIL:")
        print(stdout.read().decode('utf-8'))
        
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep -E 'deploy|rsync' | grep -v grep || echo 'no process'")
        print("RUNNING DEPLOY PROCESSES:")
        print(stdout.read().decode('utf-8'))
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
