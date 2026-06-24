import { AccountFacts } from "./api/client";

// Human-readable labels for the raw enrichment codes the pipeline emits.
export const INTENT_LABEL: Record<string, string> = {
  billing_dispute: "Billing dispute",
  late_fee: "Late fee",
  payment_arrangement: "Payment arrangement",
  refund: "Refund request",
  autopay_issue: "Autopay issue",
  plan_inquiry: "Plan change",
  cancellation_risk: "Cancellation risk",
  billing_inquiry: "Billing inquiry",
};

export function intentLabel(code?: string | null): string {
  if (!code) return "—";
  return INTENT_LABEL[code] ?? code.replace(/_/g, " ");
}

export type Priority = "high" | "medium" | "low";

export interface Recommendation {
  title: string;
  detail: string;
  priority: Priority;
}

const money = (v: unknown): string => {
  const n = typeof v === "string" ? parseFloat(v) : (v as number);
  return Number.isFinite(n) ? `$${n.toFixed(2)}` : "$0.00";
};

function overdueInvoice(facts?: AccountFacts | null) {
  return facts?.invoices?.find((i) => i.status === "overdue");
}

/**
 * Translate the enrichment signals + live account facts into a concrete,
 * fact-grounded action the agent can take right now. This is the whole point of
 * the cockpit: not "offer_fee_waiver" but "waive the $40 late fee on INV-90071".
 */
export function recommend(
  nba: string | undefined,
  sentiment: string | undefined,
  facts?: AccountFacts | null
): Recommendation {
  const cust = facts?.customer ?? {};
  const od = overdueInvoice(facts);
  const negative = sentiment === "negative";

  switch (nba) {
    case "escalate_retention_offer":
      return {
        title: "Escalate with a retention offer",
        detail: `${cust.full_name ?? "Customer"} is a ${cust.tenure_months ?? "?"}-month ${
          cust.plan ?? ""
        } customer flagged ${cust.status ?? "at risk"}. Loop in retention and lead with a loyalty credit or plan discount before they ask to cancel.`,
        priority: "high",
      };
    case "offer_fee_waiver":
      return {
        title: od
          ? `Offer to waive the ${money(od.late_fee)} late fee on ${od.invoice_id}`
          : "Offer to waive the late fee",
        detail: od
          ? `${od.invoice_id} (${od.period}) is overdue — ${money(od.amount)} + ${money(
              od.late_fee
            )} late fee, due ${od.due_date}. Waiving the fee resolves the dispute and protects CSAT.`
          : "Acknowledge the late fee and offer a one-time goodwill waiver to de-escalate.",
        priority: negative ? "high" : "medium",
      };
    case "process_refund":
      return {
        title: "Process a refund",
        detail: od
          ? `Validate the disputed charge on ${od.invoice_id} (${money(
              od.amount
            )}) and issue a refund or credit to the account.`
          : "Validate the disputed charge and issue a refund or account credit.",
        priority: negative ? "high" : "medium",
      };
    case "set_up_payment_plan": {
      const overdueAmt = facts?.summary?.overdue_amount;
      return {
        title: "Set up a payment plan",
        detail: overdueAmt
          ? `Offer to split the ${money(overdueAmt)} overdue balance into instalments and re-enable autopay to prevent future late fees.`
          : "Offer to split the outstanding balance into instalments to keep the account current.",
        priority: "medium",
      };
    }
    default:
      return {
        title: "Continue assisting — no escalation needed",
        detail:
          "Sentiment is steady and no billing risk detected. Answer the question, confirm next steps, and close warmly.",
        priority: "low",
      };
  }
}

/** Queue priority from the conversation signals alone (no account fetch needed). */
export function callPriority(
  nba: string | undefined,
  disposition: string | undefined,
  sentiment: string | undefined
): Priority {
  if (nba === "escalate_retention_offer" || disposition === "escalated" || sentiment === "negative")
    return "high";
  if (
    nba === "offer_fee_waiver" ||
    nba === "process_refund" ||
    nba === "set_up_payment_plan" ||
    disposition === "follow_up"
  )
    return "medium";
  return "low";
}

export const PRIORITY_RANK: Record<Priority, number> = { high: 0, medium: 1, low: 2 };
