import type { CustomerWithIssue } from "../api/client";
import { uiCopy } from "../i18n";

export type CustomerIssueTag = {
  id: string;
  label: string;
  warn?: boolean;
};

export function customerIssueTags(
  customer: CustomerWithIssue | null | undefined,
  copy: ReturnType<typeof uiCopy>
): CustomerIssueTag[] {
  if (!customer) return [];

  const tags: CustomerIssueTag[] = [];
  if (customer.customer_status === "at_risk") {
    tags.push({ id: "at_risk", label: copy.issueTagAtRisk, warn: true });
  }
  if ((customer.overdue_invoice_count ?? 0) > 0) {
    tags.push({ id: "overdue", label: copy.issueTagOverdueExposure, warn: true });
  }
  if (customer.autopay_enabled === false) {
    tags.push({ id: "autopay_off", label: copy.issueTagAutopayOff });
  }
  if ((customer.recent_declined_payments ?? 0) > 0) {
    tags.push({ id: "declined", label: copy.issueTagDeclinedPayments, warn: true });
  }

  if (tags.length === 0 && customer.rationale) {
    customer.rationale.split(",").forEach((part, index) => {
      const label = part.trim();
      if (!label) return;
      const warn =
        /overdue|declined|at-risk|dispute/i.test(label) || label.includes("ค้าง") || label.includes("逾期");
      tags.push({ id: `rationale-${index}`, label, warn });
    });
  }

  return tags;
}
