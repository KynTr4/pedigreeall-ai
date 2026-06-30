import sys
import paramiko
import os
import glob
from stat import S_ISDIR, S_ISREG

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"
REMOTE_TMP_DIR = "/tmp/deploy_at_yaris"
APP_DIR = "/opt/at_yaris_tahmini"

def get_latest_bundle():
    bundles = glob.glob('dist/at_yaris_tahmini_vps_with_web_*.tar.gz')
    if not bundles:
        raise Exception("No bundle found in dist/")
    latest = max(bundles, key=os.path.getctime)
    return latest

def main():
    bundle_path = get_latest_bundle()
    bundle_name = os.path.basename(bundle_path)
    print(f"Deploying bundle: {bundle_path}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        print("Connected via SSH")
        
        # Create tmp directory
        ssh.exec_command(f"mkdir -p {REMOTE_TMP_DIR}")
        
        remote_bundle_path = f"{REMOTE_TMP_DIR}/{bundle_name}"
        sftp = ssh.open_sftp()
        try:
            stat = sftp.stat(remote_bundle_path)
            if stat.st_size == os.path.getsize(bundle_path):
                print("Bundle already uploaded and size matches, skipping upload.")
                upload_needed = False
            else:
                upload_needed = True
        except IOError:
            upload_needed = True
            
        if upload_needed:
            print(f"Uploading bundle to {remote_bundle_path}...")
            sftp.put(bundle_path, remote_bundle_path)
            print("Upload complete")
        sftp.close()
        
        # Execute deployment commands (redirect output to avoid deadlock)
        commands = [
            "#!/bin/bash",
            "exec > /tmp/deploy.log 2>&1",
            "set -x",
            "STAMP=$(date +%Y%m%d_%H%M%S)",
            "BACKUP_DIR=/var/backups/at_yaris_tahmini/deploy_backup_${STAMP}",
            "mkdir -p $BACKUP_DIR",
            "rsync -avz --exclude='pedigreeall_progress.db*' --exclude='*.db*' --exclude='output/final_benter_dataset.*' --exclude='logs/*' --exclude='backups/*' /opt/at_yaris_tahmini/ $BACKUP_DIR/",
            "echo 'Backup created at '$BACKUP_DIR",
            
            f"cd {REMOTE_TMP_DIR} && tar -xzf {bundle_name}",
            
            "rsync -avz --exclude='pedigreeall_progress.db*' --exclude='.env' --exclude='output/final_benter_dataset.*' "
            f"{REMOTE_TMP_DIR}/at_yaris_tahmini/ {APP_DIR}/",
            
            f"cp -v {APP_DIR}/deploy/systemd/*.service /etc/systemd/system/ || true",
            f"cp -v {APP_DIR}/deploy/systemd/*.timer /etc/systemd/system/ || true",
            "systemctl daemon-reload",
            
            "chown -R at_yaris:at_yaris /opt/at_yaris_tahmini /var/log/at_yaris_tahmini /var/backups/at_yaris_tahmini",
            "chmod 600 /opt/at_yaris_tahmini/.env || true",
            
            "systemctl restart at-yaris-web.service",
            "systemctl restart at-yaris-results-update.timer",
            "systemctl enable --now at-yaris-race-freeze.timer || true",
            "systemctl enable --now at-yaris-results-update.timer || true",
            
            f"rm -rf {REMOTE_TMP_DIR}/at_yaris_tahmini",
            "echo 'DEPLOYMENT SUCCESSFUL'"
        ]
        
        full_command = "\n".join(commands)
        print("Running deployment script on VPS...")
        stdin, stdout, stderr = ssh.exec_command("bash")
        stdin.write(full_command)
        stdin.close()
        
        exit_status = stdout.channel.recv_exit_status()
        
        # Fetch log
        stdin, stdout, stderr = ssh.exec_command("cat /tmp/deploy.log")
        print("Deployment Log:")
        print(stdout.read().decode('utf-8'))
        
        print(f"Deployment finished with status: {exit_status}")
        
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
