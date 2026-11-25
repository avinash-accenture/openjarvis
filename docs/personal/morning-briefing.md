# Morning briefing (Phase 2)

A daily, fully local briefing of the last 24 hours of mail and the next few
days of calendar, spoken by macOS `say` (Samantha) and saved with a written
report you can check.

## How it works

1. **Collect** (`digest_collect`): every Inbox message from the last 24 hours in
   all five Apple Mail accounts, plus calendar events for today and the next two
   days, plus birthdays and anniversaries for the next 14 days.
2. **Classify** each email with explicit rules (`mail_classifier`, v1.1).
   Marketing is excluded; everything else is ranked:
   action required → time sensitive → personal → other review items →
   completed → lower priority.
3. **Select** the top 15 across all accounts. Excluded and unselected mail is
   counted, not dropped silently.
4. **Summarise** (optional): the local model summarises up to five generic
   emails that have a body. A summary is kept only if every number and name in
   it appears in the email; otherwise the briefing uses a labelled preview.
5. **Build** the spoken text from data and fixed templates only (no model
   writes the briefing). Each important email is read individually: sender,
   time, subject, then a *Summary*, a *Preview*, or the classification note.
   Lower-priority mail is named compactly by sender. Possible duplicate calendar
   entries are pointed out, never merged.
6. **Fit** the listening budget (5 minutes ≈ 630 words at Samantha 0.72): shorten
   previews, then trim low-priority detail, then move the lowest-ranked emails to
   the written report, saying how many.
7. **Speak and save**: audio plus `~/.openjarvis/digests/briefing-YYYY-MM-DD-HHMM.md`
   (mode 600) with coverage per account, every selected email with its
   category, evidence and Summary/Preview label, the calendar, and validation
   results.

## Configuration (`~/.openjarvis/config.toml`)

```toml
[digest]
schedule = "10 9 * * *"
timezone = "America/New_York"
sections = ["messages", "calendar"]
honorific = "sir"
voice_id = "Samantha"
voice_speed = 0.72
tts_backend = "macos_say"
max_minutes = 5.0          # spoken-length budget
summaries = 5              # grounded local summaries; 0 disables
special_days_ahead = 14    # birthday / anniversary lookahead

[digest.messages]
sources = ["apple_mail"]
max_items = 15             # emails selected across all accounts

[digest.calendar]
sources = ["apple_calendar"]
```

Note `max_items = 15`: the earlier config had 10.

## Validating on the Mac

Run it by hand first, several mornings in a row:

```bash
jarvis briefing run --play
```

Then open the report it prints and check:

- **Mail coverage**: all five accounts listed, none with an error, and the
  "With body" column close to "Messages". Low body coverage means Apple Mail has
  not downloaded those messages; the briefing says so rather than guessing.
- **Selected email**: the important messages are at the top; nothing
  important is in "Not selected by rank"; categories look right. Anything that
  lands in `REVIEW / unclassified` is a candidate for a new rule.
- **Summary vs Preview**: summaries match the email; previews are the email's
  own first words.
- **Calendar**: events and times match Calendar; "Possible duplicates" are
  genuinely worth a look; birthdays and anniversaries are present.
- **Length**: under five minutes.
- **Validation**: no ⚠️ lines, or ones you understand.

`jarvis briefing status` shows the last run at any time.

## Scheduling

Once manual runs are consistently right:

```bash
cd ~/Documents/OpenJarvis && jarvis briefing install-schedule
```

This installs two LaunchAgents:

- `com.openjarvis.morning-briefing`: daily at 9:10 (the Mac's local time). If
  the Mac is asleep then, macOS runs it on wake.
- `com.openjarvis.morning-briefing.catch-up`: at login, runs the briefing if the
  Mac was off at 9:10, today's briefing has not succeeded, and it is before noon.

Each run waits up to three minutes for Ollama, writes
`~/.openjarvis/digests/last_run.json`, posts a macOS notification (generic text
only, never mail content) on success or failure, and logs to
`~/.openjarvis/logs/briefing.log`. A lock stops overlapping runs. Remove with
`jarvis briefing uninstall-schedule`.

Test it: schedule a time a few minutes ahead with `--time HH:MM`, confirm it
runs; put the Mac to sleep across the time and confirm it runs on wake; shut
down across the time and confirm the login catch-up; stop Ollama and confirm a
failure notification and a clear entry in `jarvis briefing status`.

## Known limits

- The connector can only read what Apple Mail has synchronised locally.
- Account IDs, the Gmail All Mail mailbox (50) and the Gmail Inbox label (72)
  in `connectors/apple_mail.py` are specific to this Mac's Mail database.
- Calendar birthdays must exist as Calendar events (e.g. a "Birthdays"
  calendar or titles containing "birthday"/"anniversary"); Contacts is not read.
