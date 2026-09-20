#!/usr/bin/env python3
"""Keep the deployer's public IPv4 on a workspace IP allow list.

The IP Access Lists API lives on the workspace host and is itself enforced by
those lists. While this machine is still allowed, the installer can add the
current /32 to an installer-owned ALLOW list (additive; other lists are left
alone). If the machine is already blocked, Databricks will reject the same API
and an admin must apply the change from an allowed network.
"""
from __future__ import annotations

import ipaddress
import json
import os
import sys
import urllib.request
from collections.abc import Iterable

LIST_LABEL = "genie-voice-deploy"


def _log(message: str) -> None:
    print(f"[app-deploy] {message}", flush=True)


def public_ipv4() -> str:
    urls = (
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://ipv4.icanhazip.com",
    )
    last_error: Exception | None = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                candidate = response.read().decode("utf-8").strip()
            ipaddress.IPv4Address(candidate)
            return candidate
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"could not determine public IPv4: {last_error}")


def cidr_covers(entry: str, ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(entry, strict=False)
    except ValueError:
        return False


def _list_type(item: object) -> str:
    raw = getattr(item, "list_type", "")
    raw = getattr(raw, "value", raw)
    return str(raw or "").upper()


def is_blocked_by_deny(lists: Iterable[object], ip: str) -> object | None:
    for item in lists:
        if _list_type(item) != "BLOCK":
            continue
        if getattr(item, "enabled", True) is False:
            continue
        for entry in getattr(item, "ip_addresses", None) or []:
            if cidr_covers(str(entry), ip):
                return item
    return None


def already_allowed(lists: Iterable[object], ip: str) -> bool:
    allow_lists = [
        item
        for item in lists
        if _list_type(item) == "ALLOW"
        and getattr(item, "enabled", True) is not False
    ]
    if not allow_lists:
        # Feature enabled with no allow lists still admits every IP.
        return True
    return any(
        cidr_covers(str(entry), ip)
        for item in allow_lists
        for entry in (getattr(item, "ip_addresses", None) or [])
    )


def find_owned_allow_list(lists: Iterable[object]):
    for item in lists:
        if (
            getattr(item, "label", None) == LIST_LABEL
            and _list_type(item) == "ALLOW"
        ):
            return item
    return None


def merged_addresses(existing: Iterable[str] | None, ip: str) -> list[str]:
    cidr = f"{ip}/32"
    seen: list[str] = []
    for entry in list(existing or []) + [cidr]:
        value = str(entry).strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _is_ip_acl_error(exc: BaseException) -> bool:
    text = str(exc)
    return "IP ACL" in text or "ip access list" in text.lower()


def _ip_access_enabled(client) -> bool:
    try:
        conf = client.workspace_conf.get_status(keys="enableIpAccessLists") or {}
    except Exception as exc:  # noqa: BLE001
        _log(f"could not read enableIpAccessLists ({exc}); skipping IP-list update")
        return False
    value = str(conf.get("enableIpAccessLists") or "").strip().lower()
    return value in {"true", "1", "yes"}


def ensure_current_ip_allowed() -> str:
    ip = public_ipv4()
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import PermissionDenied
    from databricks.sdk.service.settings import ListType

    client = WorkspaceClient()
    try:
        me = client.current_user.me()
    except PermissionDenied as exc:
        if _is_ip_acl_error(exc):
            raise SystemExit(
                f"Source IP {ip} is blocked by the workspace IP access list.\n"
                "The IP Access Lists API is on the same workspace host, so this "
                "installer cannot add the IP from the blocked machine.\n"
                "From an allowed network (VPN / office / already-allow-listed "
                "admin), run:\n"
                f"  databricks ip-access-lists create --json "
                f"'{json.dumps({'label': LIST_LABEL, 'list_type': 'ALLOW', 'ip_addresses': [f'{ip}/32']})}'\n"
                "then rerun ./deploy_app.sh from this machine."
            ) from exc
        raise SystemExit(
            "Not authenticated. Run: databricks auth login"
            + (f" --profile {os.environ['DATABRICKS_CONFIG_PROFILE']}"
               if os.environ.get("DATABRICKS_CONFIG_PROFILE") else "")
        ) from exc

    user = getattr(me, "user_name", None) or getattr(me, "userName", None) or "ok"
    _log(f"workspace reachable as {user} from {ip}")
    if os.environ.get("UPDATE_IP_ACCESS_LIST", "1").strip() in {"0", "false", "no"}:
        return ip
    if not _ip_access_enabled(client):
        _log("workspace IP access lists are disabled; no allow-list update needed")
        return ip

    try:
        lists = list(client.ip_access_lists.list())
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN: cannot list IP access lists ({exc}); continuing while the workspace is reachable")
        return ip

    blocked = is_blocked_by_deny(lists, ip)
    if blocked is not None:
        raise SystemExit(
            f"Source IP {ip} is on BLOCK list '{getattr(blocked, 'label', None)}' "
            f"({getattr(blocked, 'list_id', None)}). The installer will not remove "
            "block lists; update that list from an admin session and rerun."
        )
    if already_allowed(lists, ip):
        _log(f"IP {ip} is already admitted by an existing ALLOW list")
        return ip

    owned = find_owned_allow_list(lists)
    addresses = merged_addresses(getattr(owned, "ip_addresses", None) if owned else None, ip)
    try:
        if owned is None:
            client.ip_access_lists.create(
                label=LIST_LABEL,
                list_type=ListType.ALLOW,
                ip_addresses=addresses,
            )
            _log(f"created ALLOW list '{LIST_LABEL}' with {ip}/32")
        else:
            client.ip_access_lists.update(
                ip_access_list_id=str(owned.list_id),
                enabled=True,
                ip_addresses=addresses,
                label=LIST_LABEL,
                list_type=ListType.ALLOW,
            )
            _log(f"updated ALLOW list '{LIST_LABEL}' to include {ip}/32")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN: could not update IP access lists ({exc}); workspace is still reachable from here")
    return ip


def main() -> None:
    ensure_current_ip_allowed()


if __name__ == "__main__":
    main()
