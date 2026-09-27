---
title: Quillfeather Labs - Postmortem for Incident INC-2291 (Ledgerlight payments outage)
---

## Summary

Incident INC-2291 was a SEV1 outage of Ledgerlight payments on August 14, 2026. From 2:12 PM to 2:59 PM Eastern Time, a total of 47 minutes, customers could not run payment reconciliation. About 1,240 customers were affected. No customer data was lost.

## Root cause

The root cause was an expired TLS certificate on Tollbooth, the internal payments gateway. The certificate had been renewed manually in 2025, and the renewal was never added to the automated certificate rotation system. When the certificate expired, every connection from Ledgerlight to Tollbooth was rejected.

## Timeline (Eastern Time)

- 2:12 PM - Error rates on payment reconciliation jump to 100%.
- 2:14 PM - Beacon pages the primary on-call engineer, Keiko Almeida.
- 2:19 PM - Keiko declares a SEV1 and becomes incident commander.
- 2:31 PM - The team finds certificate errors in the Tollbooth logs.
- 2:52 PM - A new certificate is issued and deployed to Tollbooth.
- 2:59 PM - Error rates return to normal and the incident is resolved.

## What went well

The page was acknowledged within 2 minutes. The status page was updated every 30 minutes, as required for SEV1 incidents.

## What went wrong

The expired certificate was not covered by monitoring, so there was no warning before it expired. It took 17 minutes to find the root cause because the Ledgerlight error messages said "gateway unavailable" instead of naming the certificate problem.

## Action items

1. Add every Tollbooth certificate to automated rotation. Owner: Keiko Almeida. Due: September 5, 2026. Status: done.
2. Alert 30 days and again 7 days before any certificate expires. Owner: Rafael Duarte. Due: September 19, 2026. Status: done.
3. Show the real reason for gateway errors in Ledgerlight logs. Owner: Sun-hee Park. Due: October 10, 2026. Status: in progress.
4. Offer affected customers a 10% credit on their September invoice. Owner: Adaeze Mbeki. Status: done.
