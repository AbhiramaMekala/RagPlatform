---
title: Quillfeather Labs - Engineering On-Call and Deployment Rules
---

## On-call rotation

Every backend engineer at Quillfeather Labs joins the on-call rotation after completing their first 90 days. Each on-call shift lasts one week. The handoff happens every Tuesday at 10:00 AM Eastern Time, during the Tuesday operations meeting.

Pages are sent through Beacon. The primary on-call engineer must acknowledge a page within 5 minutes. If the page is not acknowledged, Beacon escalates to the secondary on-call engineer, and after 15 minutes to the VP of Engineering, Tomasz Wierzbicki.

On-call engineers receive an on-call stipend of $400 for each week they are on call, plus one day off after any week with a SEV1 incident.

## Severity levels

Incidents are classified into three severity levels:
- SEV1: Ledgerlight is down or customer payments are failing. Response within 15 minutes, 24 hours a day. An incident commander must be named and customers are updated on the status page every 30 minutes.
- SEV2: A major feature is broken or slow for many customers, but a workaround exists. Response within 1 hour during business hours.
- SEV3: A minor bug or a problem affecting a single customer. Handled the next business day.

Every SEV1 incident requires a written postmortem within 5 business days.

## Deployment rules

Ledgerlight ships on the Orbit release train every second Wednesday. Hotfixes can be deployed at any time with approval from the on-call engineer.

Deployments are frozen on Fridays after 2:00 PM Eastern Time and during Quiet Weeks. Deployments to Tollbooth, the payments gateway, always need two approving code reviews and must happen between 10:00 AM and 3:00 PM Eastern Time.

## Code freeze

Before a major Ledgerlight version, there is a code freeze of 10 business days. During a code freeze only bug fixes can be merged into the release branch.
