"""
Health Check Module for Terminal Chatbot
Provides health check endpoints and system status monitoring.
"""

import os
import sys
import time
import platform
import psutil
from datetime import datetime
from typing import Dict, Optional

from logger import get_logger

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = get_logger()

# Track startup time
_startup_time = datetime.now()


def get_system_info() -> Dict:
    """Get basic system information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid()
    }


def get_resource_usage() -> Dict:
    """Get current resource usage."""
    try:
        process = psutil.Process()
        memory = process.memory_info()
        return {
            "memory_mb": round(memory.rss / (1024 * 1024), 2),
            "memory_percent": round(process.memory_percent(), 2),
            "cpu_percent": round(process.cpu_percent(interval=0.1), 2),
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        }
    except Exception as e:
        return {"error": str(e)}


def check_disk_space(path: str = ".") -> Dict:
    """Check available disk space."""
    try:
        usage = psutil.disk_usage(path)
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": usage.percent
        }
    except Exception as e:
        return {"error": str(e)}


def check_openai_api() -> Dict:
    """Check OpenAI API connectivity."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "unconfigured", "error": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=5)
        # Just check we can create a client - don't make actual API call
        return {"status": "configured", "key_prefix": api_key[:8] + "..."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_database() -> Dict:
    """Check database connectivity."""
    try:
        from database import Database, POSTGRES_AVAILABLE
        if not POSTGRES_AVAILABLE:
            return {"status": "unavailable", "error": "psycopg2 not installed"}

        db = Database.get_instance()
        return db.health_check()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_storage() -> Dict:
    """Check S3 storage connectivity (required for production)."""
    try:
        from storage import get_storage, S3_ENABLED, S3_AVAILABLE
        if not S3_ENABLED:
            return {"status": "error", "error": "S3_ENABLED must be true for production"}
        if not S3_AVAILABLE:
            return {"status": "error", "error": "boto3 not installed"}

        storage = get_storage()
        return storage.health_check()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_uptime() -> Dict:
    """Get application uptime."""
    uptime = datetime.now() - _startup_time
    return {
        "started_at": _startup_time.isoformat(),
        "uptime_seconds": int(uptime.total_seconds()),
        "uptime_human": str(uptime).split('.')[0]  # Remove microseconds
    }


def health_check(include_details: bool = False) -> Dict:
    """
    Perform comprehensive health check.

    Args:
        include_details: Include detailed system information

    Returns:
        Health check results
    """
    start_time = time.time()

    # Core checks
    checks = {
        "openai_api": check_openai_api(),
        "database": check_database(),
        "storage": check_storage(),
        "disk": check_disk_space()
    }

    # Determine overall status
    critical_services = ["openai_api"]
    all_healthy = True
    degraded = False

    for service, result in checks.items():
        status = result.get("status", "unknown")
        if status in ["error", "unhealthy"]:
            if service in critical_services:
                all_healthy = False
            else:
                degraded = True

    if all_healthy and not degraded:
        overall_status = "healthy"
    elif all_healthy and degraded:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    result = {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "uptime": get_uptime(),
        "checks": checks,
        "response_time_ms": round((time.time() - start_time) * 1000, 2)
    }

    if include_details:
        result["system"] = get_system_info()
        result["resources"] = get_resource_usage()

    return result


def liveness_check() -> Dict:
    """
    Simple liveness check (is the application running?).
    Used by Kubernetes/Docker health probes.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


def readiness_check() -> Dict:
    """
    Readiness check (is the application ready to serve requests?).
    Used by Kubernetes/Docker readiness probes.
    """
    # Check critical services
    api_status = check_openai_api()
    storage_status = check_storage()

    issues = []
    if api_status.get("status") != "configured":
        issues.append("OpenAI API not configured")
    if storage_status.get("status") not in ["healthy"]:
        issues.append(f"S3 storage: {storage_status.get('error', 'unavailable')}")

    if not issues:
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }

    return {
        "status": "not_ready",
        "timestamp": datetime.now().isoformat(),
        "reason": "; ".join(issues)
    }


def print_health_status():
    """Print health status to console."""
    result = health_check(include_details=True)

    print("\n" + "=" * 60)
    print("HEALTH CHECK REPORT")
    print("=" * 60)

    print(f"\nOverall Status: {result['status'].upper()}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Response Time: {result['response_time_ms']}ms")

    print(f"\nUptime: {result['uptime']['uptime_human']}")
    print(f"Started: {result['uptime']['started_at']}")

    print("\n--- Service Checks ---")
    for service, check in result['checks'].items():
        status = check.get('status', 'unknown')
        icon = "[OK]" if status in ['healthy', 'configured', 'local'] else "[!!]"
        print(f"  {icon} {service}: {status}")
        if 'error' in check:
            print(f"      Error: {check['error']}")

    if 'resources' in result:
        print("\n--- Resource Usage ---")
        res = result['resources']
        if 'error' not in res:
            print(f"  Memory: {res.get('memory_mb', 'N/A')} MB ({res.get('memory_percent', 'N/A')}%)")
            print(f"  CPU: {res.get('cpu_percent', 'N/A')}%")
            print(f"  Threads: {res.get('threads', 'N/A')}")

    if 'system' in result:
        print("\n--- System Info ---")
        sys_info = result['system']
        print(f"  Platform: {sys_info.get('platform', 'N/A')}")
        print(f"  Python: {sys_info.get('python_version', 'N/A').split()[0]}")
        print(f"  PID: {sys_info.get('pid', 'N/A')}")

    print("\n" + "=" * 60)

    return result


if __name__ == "__main__":
    print_health_status()
