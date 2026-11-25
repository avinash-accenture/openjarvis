# Remote access from iPhone and iPad (Phase 3)

OpenJarvis keeps listening on `127.0.0.1` only. Tailscale Serve gives the Mac a
private HTTPS address on your tailnet (`https://<mac>.<tailnet>.ts.net`) and
forwards it to the local server. Only devices signed in to your tailnet can
resolve or reach that address; nothing is opened on your router and nothing is
exposed to the internet.

```
iPhone / iPad (Tailscale app)
        │  WireGuard tunnel, HTTPS to <mac>.<tailnet>.ts.net
        ▼
Mac: Tailscale Serve ──► http://127.0.0.1:8000 (jarvis serve, API key required)
```

Network privacy (Tailscale) and application authentication (the API key) are
separate controls, and both stay on. Requests forwarded by Tailscale arrive
from `127.0.0.1`, which is exactly why the server requires the key even on
loopback.

## What the server enforces for remote use

| Control | Behaviour |
| --- | --- |
| Loopback only | `jarvis serve` refuses any other bind address. |
| API key | Required on every `/v1`, `/api`, metrics and docs route, and on WebSockets. |
| Host check | Only `localhost`, `127.0.0.1`, `::1` and `*.ts.net` names are served (HTTP 421 otherwise). Blocks DNS-rebinding attacks from web pages. Extra names: `[server] allowed_hosts`. |
| Tailscale user (optional) | `[server] tailscale_allowed_logins = ["you@example.com"]`: forwarded requests must come from these Tailscale accounts. |
| No caching | API responses carry `Cache-Control: no-store`, so mail and calendar data are not kept in the phone's browser cache. |
| Content Security Policy | Scripts only from the server itself (no inline script or eval); the page cannot connect anywhere except the server. |
| No public exposure | `jarvis tunnel` (Cloudflare) refuses to run; `jarvis remote check` fails if Tailscale Funnel is on. |
| No analytics prompt | The web app does not offer analytics or the leaderboard on a local-only server. |

## Setup

### 1. Mac

1. Install Tailscale from <https://tailscale.com/download/mac> and sign in.
   This Mac runs macOS 12.7; check the download page for the version that
   supports Monterey.
2. In the Tailscale admin console (DNS page) turn on **MagicDNS** and
   **HTTPS certificates**.
3. Keep the server running in the background, and point Tailscale Serve at it:

   ```bash
   cd ~/Documents/OpenJarvis && jarvis remote install-server
   ```

   ```bash
   jarvis remote setup --apply
   ```

   The second command runs
   `tailscale serve --bg --https=443 http://127.0.0.1:8000`.
4. Check everything:

   ```bash
   jarvis remote check
   ```

   It fails if Tailscale is not connected, Serve does not forward to the local
   server, Funnel is on, the API key is missing, or the HTTPS address does not
   require the key.
5. The Mac must be awake to answer. In System Preferences → Battery → Power
   Adapter, turn on "Prevent computer from sleeping automatically when the
   display is off" (or run `caffeinate -s` while plugged in).

### 2. iPhone / iPad

1. Install Tailscale from the App Store and sign in with the same account.
2. In Safari open `https://<mac>.<tailnet>.ts.net` (shown by `jarvis remote check`).
3. Open **Settings** in OpenJarvis and paste the API key into **API key**. The
   key is in `~/.openjarvis/config.toml` under `[server.auth]`; copy it over a
   trusted channel (for example AirDrop to yourself), never in a screenshot or
   chat.
4. Share → **Add to Home Screen** to use it like an app.

### 3. Tailnet access rule (recommended)

If anyone else is on your tailnet, limit who can reach the Mac. In the admin
console's access controls, tag the Mac (for example `tag:jarvis`) and allow
only your own account to its HTTPS port:

```jsonc
{
  "tagOwners": { "tag:jarvis": ["autogroup:admin"] },
  "grants": [
    { "src": ["you@example.com"], "dst": ["tag:jarvis"], "ip": ["443"] }
  ]
}
```

Merge this with your existing policy; replace `you@example.com` with your
Tailscale login. Add the same login to `[server] tailscale_allowed_logins` for
a second, application-level check.

## Testing (Phase 3 definition of done)

- From the phone on cellular data (Wi-Fi off), with Tailscale on: the app loads,
  chat streams, and the dashboard shows data.
- Turn Tailscale off on the phone: the address no longer resolves.
- From any device not on the tailnet: the address does not resolve.
- Run `scripts/verify_phase1_security.sh` on the Mac: all checks pass,
  including section 6 (remote access).
- `jarvis remote check` reports no Funnel.

## If a device is lost

1. Remove it in the Tailscale admin console (Machines → … → Remove).
2. Rotate the key: `jarvis auth create-key`, then restart the server
   (`jarvis remote install-server` reloads it).
3. Enter the new key on your remaining devices.
