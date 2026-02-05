"""
License Manager - Client-side license verification.

Handles:
- Hardware ID generation
- License verification against server
- Local cache for offline operation (24h)
- Feature checking
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests

# ============== Configuration ==============

# Import from central config
try:
    from .config import LICENSE_SERVER_URL, LICENSE_DEV_MODE
except ImportError:
    try:
        # Intento con import absoluto (para apps independientes)
        from shared.config import LICENSE_SERVER_URL, LICENSE_DEV_MODE
    except ImportError:
        # Fallback final
        LICENSE_SERVER_URL = os.getenv(
            "LICENSE_SERVER_URL",
            "https://tu-dominio.com/api/v1/license"
        )
        # TEMPORAL: Licencias desactivadas - cambiar a False para activar
        LICENSE_DEV_MODE = True

# Cache settings
CACHE_FILE = Path.home() / ".embedding_dashboard_license.json"
CACHE_HOURS = 24

# Request timeout
REQUEST_TIMEOUT = 10  # seconds

# Headers para evitar bloqueo de WAF
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


# ============== Data Classes ==============

@dataclass
class LicenseInfo:
    """License information from server or cache."""
    valid: bool
    license_key: str = ""
    email: str = ""
    plan: str = ""
    features: List[str] = field(default_factory=list)
    expires: str = ""
    days_remaining: int = 0
    error: str = ""
    from_cache: bool = False
    cache_expires: Optional[datetime] = None


# ============== Trial Features ==============

TRIAL_FEATURES = [
    "hub",
    "csv_limited",  # Max 100 rows
    "semantic_tools",
]


# ============== License Manager ==============

class LicenseManager:
    """
    Manages license verification and caching.

    Usage:
        manager = LicenseManager()

        # Check if licensed
        license_info = manager.verify_license("EMB-XXXX-XXXX-XXXX-XXXX")
        if license_info.valid:
            # Full access
            pass
        else:
            # Trial mode
            pass

        # Check specific feature
        if manager.is_feature_enabled("fanout"):
            # Show fanout feature
            pass
    """

    def __init__(self):
        """Initialize license manager."""
        self.hardware_id = self._get_hardware_id()
        self._current_license: Optional[LicenseInfo] = None

    def _get_hardware_id(self) -> str:
        """
        Generate a unique hardware ID based on machine characteristics.

        Combines:
        - MAC address
        - Hostname
        - Platform info
        """
        components = []

        # MAC address
        try:
            mac = uuid.getnode()
            components.append(str(mac))
        except Exception:
            components.append("no-mac")

        # Hostname
        try:
            components.append(socket.gethostname())
        except Exception:
            components.append("no-hostname")

        # Platform
        components.append(platform.system())
        components.append(platform.machine())

        # Create hash
        combined = "-".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _get_machine_name(self) -> str:
        """Get human-readable machine name."""
        try:
            hostname = socket.gethostname()
            system = platform.system()
            return f"{hostname} ({system})"
        except Exception:
            return "Unknown"

    def _load_cache(self) -> Optional[dict]:
        """Load cached license data."""
        if not CACHE_FILE.exists():
            return None

        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if cache is for this hardware
            if data.get("hardware_id") != self.hardware_id:
                return None

            # Check if cache is expired
            cache_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.now() - cache_time > timedelta(hours=CACHE_HOURS):
                return None

            return data

        except Exception:
            return None

    def _save_cache(self, license_info: LicenseInfo, license_key: str) -> None:
        """Save license data to cache."""
        try:
            data = {
                "hardware_id": self.hardware_id,
                "license_key": license_key,
                "valid": license_info.valid,
                "email": license_info.email,
                "plan": license_info.plan,
                "features": license_info.features,
                "expires": license_info.expires,
                "days_remaining": license_info.days_remaining,
                "cached_at": datetime.now().isoformat(),
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Cache save failure is not critical

    def _clear_cache(self) -> None:
        """Clear cached license data."""
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
        except Exception:
            pass

    def verify_license(self, license_key: str) -> LicenseInfo:
        """
        Verify a license key against the server.

        Args:
            license_key: The license key to verify

        Returns:
            LicenseInfo with verification result
        """
        # Dev mode: todas las features habilitadas sin servidor
        if LICENSE_DEV_MODE:
            dev_license = LicenseInfo(
                valid=True,
                license_key="DEV-MODE",
                email="dev@localhost",
                plan="pro",
                features=["hub", "csv", "semantic_tools", "keywords", "linking",
                         "positions", "relations", "content_plan", "fanout",
                         "api", "export", "unlimited_projects"],
                expires="2099-12-31",
                days_remaining=9999,
            )
            self._current_license = dev_license
            return dev_license

        # Try server verification first
        try:
            # Usar GET para evitar bloqueo de WAF en SiteGround
            response = requests.get(
                LICENSE_SERVER_URL,
                params={
                    "action": "verify",
                    "license_key": license_key,
                    "hardware_id": self.hardware_id,
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                license_info = LicenseInfo(
                    valid=data.get("valid", False),
                    license_key=data.get("license_key", ""),
                    email=data.get("email", ""),
                    plan=data.get("plan", ""),
                    features=data.get("features", []),
                    expires=data.get("expires", ""),
                    days_remaining=data.get("days_remaining", 0),
                    error=data.get("error", ""),
                )

                # Save to cache if valid
                if license_info.valid:
                    self._save_cache(license_info, license_key)

                self._current_license = license_info
                return license_info

        except requests.exceptions.RequestException:
            # Server unreachable, try cache
            pass

        # Fall back to cache
        cached = self._load_cache()
        if cached and cached.get("license_key") == license_key:
            cache_time = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            license_info = LicenseInfo(
                valid=cached.get("valid", False),
                license_key=cached.get("license_key", ""),
                email=cached.get("email", ""),
                plan=cached.get("plan", ""),
                features=cached.get("features", []),
                expires=cached.get("expires", ""),
                days_remaining=cached.get("days_remaining", 0),
                from_cache=True,
                cache_expires=cache_time + timedelta(hours=CACHE_HOURS),
            )
            self._current_license = license_info
            return license_info

        # No valid license
        self._current_license = LicenseInfo(
            valid=False,
            error="Could not verify license. Please check your internet connection.",
        )
        return self._current_license

    def activate_license(self, license_key: str, email: str) -> tuple[bool, str]:
        """
        Activate a license on this device.

        Args:
            license_key: The license key to activate
            email: Email associated with the license

        Returns:
            Tuple of (success, message)
        """
        # Dev mode: activación simulada
        if LICENSE_DEV_MODE:
            self.verify_license(license_key)  # Sets dev license
            return True, "Licencia activada (modo desarrollo)"

        try:
            # Usar GET para evitar bloqueo de WAF en SiteGround
            response = requests.get(
                LICENSE_SERVER_URL,
                params={
                    "action": "activate",
                    "license_key": license_key,
                    "hardware_id": self.hardware_id,
                    "email": email,
                    "machine_name": self._get_machine_name(),
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                message = data.get("message", "")

                if success:
                    # Verify and cache
                    self.verify_license(license_key)

                return success, message

            return False, f"Server error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

    def deactivate_license(self, license_key: str) -> tuple[bool, str]:
        """
        Deactivate the license from this device.

        Args:
            license_key: The license key to deactivate

        Returns:
            Tuple of (success, message)
        """
        try:
            # Usar GET para evitar bloqueo de WAF en SiteGround
            response = requests.get(
                LICENSE_SERVER_URL,
                params={
                    "action": "deactivate",
                    "license_key": license_key,
                    "hardware_id": self.hardware_id,
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                message = data.get("message", "")

                if success:
                    self._clear_cache()
                    self._current_license = None

                return success, message

            return False, f"Server error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

    def check_cached_license(self) -> Optional[LicenseInfo]:
        """
        Check if there's a valid cached license.

        Returns:
            LicenseInfo if cached license exists and is valid, None otherwise
        """
        # Dev mode: retornar licencia dev sin verificar cache
        if LICENSE_DEV_MODE:
            return self.verify_license("DEV-MODE")

        cached = self._load_cache()
        if not cached:
            return None

        license_key = cached.get("license_key", "")
        if not license_key:
            return None

        return self.verify_license(license_key)

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a specific feature is enabled.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is enabled, False otherwise
        """
        if self._current_license and self._current_license.valid:
            return feature in self._current_license.features

        # Trial features
        return feature in TRIAL_FEATURES

    def get_available_features(self) -> List[str]:
        """
        Get list of available features.

        Returns:
            List of enabled feature names
        """
        if self._current_license and self._current_license.valid:
            return self._current_license.features

        return TRIAL_FEATURES

    @property
    def is_licensed(self) -> bool:
        """Check if there's a valid license."""
        return self._current_license is not None and self._current_license.valid

    @property
    def current_license(self) -> Optional[LicenseInfo]:
        """Get current license info."""
        return self._current_license

    @property
    def is_trial(self) -> bool:
        """Check if running in trial mode."""
        return not self.is_licensed


# ============== Singleton Instance ==============

_license_manager: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    """Get or create the license manager singleton."""
    global _license_manager
    if _license_manager is None:
        _license_manager = LicenseManager()
    return _license_manager


# ============== Convenience Functions ==============

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return get_license_manager().is_feature_enabled(feature)


def get_available_features() -> List[str]:
    """Get list of available features."""
    return get_license_manager().get_available_features()


def is_licensed() -> bool:
    """Check if there's a valid license."""
    return get_license_manager().is_licensed


def is_trial() -> bool:
    """Check if running in trial mode."""
    return get_license_manager().is_trial
