import { useMemo, useState } from "react";
import { CallState, CustomerWithIssue, InteractionLanguage } from "../api/client";
import { PRIORITY_RANK } from "../guidance";

const SPOTLIGHT_CUSTOMER_ID = "CUST-4028";

function customerPriority(c: CustomerWithIssue) {
  if (c.issue_status !== "closed") return "high" as const;
  if ((c.overdue_invoice_count ?? 0) > 0 || c.customer_status === "at_risk") return "medium" as const;
  return "low" as const;
}

export function sortCustomersForQueue(customers: CustomerWithIssue[]) {
  const base = [...customers].sort((a, b) => {
    const pa = PRIORITY_RANK[customerPriority(a)];
    const pb = PRIORITY_RANK[customerPriority(b)];
    if (pa !== pb) return pa - pb;
    return (b.overdue_amount ?? 0) - (a.overdue_amount ?? 0);
  });
  const spotlight = base.find((c) => c.customer_id === SPOTLIGHT_CUSTOMER_ID);
  if (!spotlight) return base;
  return [spotlight, ...base.filter((c) => c.customer_id !== SPOTLIGHT_CUSTOMER_ID)];
}

export function useCustomerCall({
  customers,
  calls,
  customerId,
}: {
  customers: CustomerWithIssue[];
  calls: CallState[];
  customerId: string | null;
}) {
  const sortedCustomers = useMemo(() => sortCustomersForQueue(customers), [customers]);

  const callByCustomer = useMemo(() => {
    const map = new Map<string, CallState>();
    for (const c of calls) {
      if (c.customer_id && !map.has(c.customer_id)) map.set(c.customer_id, c);
    }
    return map;
  }, [calls]);

  const [conversationByCall, setConversationByCall] = useState<
    Record<string, Array<{ text: string; speaker?: number; language?: InteractionLanguage }>>
  >({});

  const selectedCustomer =
    sortedCustomers.find((c) => c.customer_id === customerId) ?? sortedCustomers[0] ?? null;

  const selectedCall = selectedCustomer?.call_id
    ? calls.find((c) => c.call_id === selectedCustomer.call_id) ??
      callByCustomer.get(selectedCustomer.customer_id) ??
      null
    : selectedCustomer
    ? callByCustomer.get(selectedCustomer.customer_id) ?? null
    : null;

  return {
    sortedCustomers,
    selectedCustomer,
    selectedCall,
    conversationByCall,
    appendTurn: (callId: string, turn: { text: string; speaker?: number; language?: InteractionLanguage }) =>
      setConversationByCall((prev) => ({
        ...prev,
        [callId]: [...(prev[callId] ?? []), turn],
      })),
    updateLastCustomerTurn: (
      callId: string,
      turn: { text: string; speaker?: number; language?: InteractionLanguage }
    ) =>
      setConversationByCall((prev) => {
        const rows = [...(prev[callId] ?? [])];
        for (let i = rows.length - 1; i >= 0; i -= 1) {
          if ((rows[i].speaker ?? 0) === 1) {
            rows[i] = { ...rows[i], ...turn, speaker: 1 };
            return { ...prev, [callId]: rows };
          }
        }
        return { ...prev, [callId]: [...rows, { ...turn, speaker: 1 }] };
      }),
    removeLastCustomerTurn: (callId: string) =>
      setConversationByCall((prev) => {
        const rows = [...(prev[callId] ?? [])];
        for (let i = rows.length - 1; i >= 0; i -= 1) {
          if ((rows[i].speaker ?? 0) === 1) {
            rows.splice(i, 1);
            return { ...prev, [callId]: rows };
          }
        }
        return prev;
      }),
    resetTurns: (callId: string) =>
      setConversationByCall((prev) => ({
        ...prev,
        [callId]: [],
      })),
  };
}

export { customerPriority, SPOTLIGHT_CUSTOMER_ID };
