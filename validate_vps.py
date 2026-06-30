import paramiko
import sys
import datetime
import os

VPS_HOST = "5.175.136.118"
VPS_USER = "root"
VPS_PASSWORD = "!AhXHA9YHWg4Fv9"

def run_cmd(ssh, cmd):
    print(f"Running: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    return out + err

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASSWORD, timeout=10)
        
        # Get WEB_PASSWORD from .env
        env_content = run_cmd(ssh, "cat /opt/at_yaris_tahmini/.env | grep WEB_PASSWORD || echo 'WEB_PASSWORD=testpass'")
        password = "testpass"
        for line in env_content.splitlines():
            if line.startswith("WEB_PASSWORD="):
                password = line.split("=", 1)[1].strip().strip('"').strip("'")
        
        commands = [
            "systemctl list-timers 'at-yaris-*'",
            "systemctl status at-yaris-web.service --no-pager",
            "systemctl status at-yaris-results-update.service --no-pager",
            f'curl -u admin:{password} "http://127.0.0.1:8000/api/results-refresh/status?date=2026-06-28"',
            f'curl -u admin:{password} "http://127.0.0.1:8000/api/race-day/missing-horses?date=2026-06-28"',
            f'curl -u admin:{password} "http://127.0.0.1:8000/api/bet-simulator/summary?date=2026-06-28&model=Ensemble&stake=20"'
        ]
        
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/vps_deploy_validation_{stamp}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# VPS Deploy Validation Report - {stamp}\n\n")
            for cmd in commands:
                f.write(f"## {cmd}\n")
                f.write("```\n")
                output = run_cmd(ssh, cmd)
                f.write(output)
                if not output.endswith("\n"):
                    f.write("\n")
                f.write("```\n\n")
        
        print(f"Report created at {report_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
