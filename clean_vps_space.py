import paramiko

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        
        # Delete the previous failed/old deployment backups to free up space
        print("Cleaning up old deploy backups...")
        stdin, stdout, stderr = ssh.exec_command("rm -rf /var/backups/at_yaris_tahmini/deploy_backup_*")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
        stdin, stdout, stderr = ssh.exec_command("df -h")
        print("DF AFTER CLEANUP:")
        print(stdout.read().decode('utf-8'))
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
