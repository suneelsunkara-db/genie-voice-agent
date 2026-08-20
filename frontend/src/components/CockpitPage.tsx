import { useEffect, useMemo, useState } from "react";
import {
  CustomerWithIssue,
  InteractionLanguage,
  InteractionLanguageOption,
  StatusResponse,
} from "../api/client";
import { sortCustomersForQueue, SPOTLIGHT_CUSTOMER_ID, useCustomerCall } from "../hooks/useCustomerCall";
import { customerIssueTags } from "../lib/customerIssues";
import { contentLanguage, languageLabel, uiCopy } from "../i18n";
import { CockpitSession } from "./CallList";
import {
  letterAt,
  SentientChoice,
  SentientHCol,
  SentientStep,
  SentientTopBar,
  type TopBarHighlight,
} from "./sentient/Sentient";

function CockpitLoadingHero({ status }: { status: string }) {
  return (
    <div className="sentient-cockpit sentient-cockpit-loading">
      <div className="sentient-loading">
        <span className="sentient-loading-spinner" aria-hidden="true" />
        <p className="sentient-arch-hero-status" aria-live="polite">
          {status}
        </p>
      </div>
    </div>
  );
}

export function CockpitPage({
  status,
  customers,
  customersLoading,
  customersErr,
  interactionLanguage,
  onLanguageChange,
}: {
  status: StatusResponse | null;
  customers: CustomerWithIssue[];
  customersLoading: boolean;
  customersErr: string | null;
  interactionLanguage: InteractionLanguage;
  onLanguageChange: (language: InteractionLanguage) => void;
}) {
  const copy = uiCopy(interactionLanguage);
  const [pickedCustomerId, setPickedCustomerId] = useState<string | null>(null);

  const sortedCustomers = useMemo(() => sortCustomersForQueue(customers), [customers]);
  const queueCustomers = useMemo(() => sortedCustomers.slice(0, 10), [sortedCustomers]);
  const activeCustomerId = pickedCustomerId ?? queueCustomers[0]?.customer_id ?? null;

  const { selectedCustomer, selectedCall, conversationByCall, appendTurn, updateLastCustomerTurn, removeLastCustomerTurn, resetTurns } =
    useCustomerCall({
      customers,
      calls: status?.call_states ?? [],
      customerId: activeCustomerId,
    });

  useEffect(() => {
    if (pickedCustomerId) return;
    const defaultId =
      queueCustomers.find((c) => c.customer_id === SPOTLIGHT_CUSTOMER_ID)?.customer_id ??
      queueCustomers[0]?.customer_id;
    if (defaultId) setPickedCustomerId(defaultId);
  }, [queueCustomers, pickedCustomerId]);

  const activeIssueTags = useMemo(
    () => customerIssueTags(selectedCustomer, copy),
    [selectedCustomer, copy]
  );

  // Voice-first language picker: the options are whatever the backend voice loop
  // supports end-to-end (~24, from one config-driven catalog), deduped across the
  // Chinese ASR variants. Native labels come from Intl.DisplayNames, so there's
  // no hardcoded per-language name list here.
  const languageOptions = useMemo<InteractionLanguageOption[]>(() => {
    const supported = status?.languages?.supported ?? [];
    const seen = new Set<string>();
    const opts: InteractionLanguageOption[] = [];
    for (const item of supported) {
      const code = contentLanguage(item.code);
      if (seen.has(code)) continue;
      seen.add(code);
      opts.push({ code, label: languageLabel(code, code), english_name: item.english_name });
    }
    if (opts.length === 0) {
      opts.push({ code: "en-US", label: languageLabel("en-US", "en-US") });
    }
    return opts;
  }, [status?.languages]);

  const supportedLanguageCount = languageOptions.length;

  // Value-prop highlights for the top bar. Colors map to accents in sentient.css.
  const topBarHighlights = useMemo<TopBarHighlight[]>(
    () => [
      { label: copy.hlInsights, accent: "insight" },
      { label: copy.hlResolution, accent: "resolution" },
      { label: copy.hlHoldTime, accent: "hold" },
      { label: copy.hlTokenomics, accent: "cost" },
      { label: copy.hlReasoning, accent: "reasoning" },
    ],
    [copy]
  );

  if (customersLoading && !customers.length) {
    return <CockpitLoadingHero status={copy.connectingWorkspace} />;
  }

  if (!sortedCustomers.length) {
    return (
      <SentientStep
        step={1}
        title={copy.sidebarTitle}
        description={customersErr ? `${copy.unableToLoadCustomers}: ${customersErr}` : copy.noCustomers}
        compact
      />
    );
  }

  return (
    <div className="sentient-cockpit">
      <div className="sentient-h-rail">
        <header className="sentient-h-rail-bar">
          <SentientTopBar
            contextKicker={copy.voiceStackKicker}
            contextDesc={copy.voiceStackDesc(supportedLanguageCount)}
            highlights={topBarHighlights}
            languageLabel={copy.interactionLanguage}
            options={languageOptions}
            value={interactionLanguage}
            onChange={onLanguageChange}
            languageDisabled
          />
        </header>

        <div className="sentient-h-rail-cols">
          <SentientHCol
            step={1}
            title={copy.queueTitle}
            description={copy.queueDesc}
            className="sentient-h-col-queue"
          >
            <div className="sentient-h-choices">
              {queueCustomers.map((customer, index) => (
                <SentientChoice
                  key={customer.customer_id}
                  letter={letterAt(index)}
                  label={customer.full_name ?? customer.customer_id}
                  selected={activeCustomerId === customer.customer_id}
                  onClick={() => setPickedCustomerId(customer.customer_id)}
                />
              ))}
            </div>
          </SentientHCol>

          {selectedCall ? (
            <CockpitSession
              layout="horizontal"
              call={selectedCall}
              customer={selectedCustomer}
              customerName={selectedCustomer?.full_name ?? null}
              callLabel={`${copy.call} ${selectedCall.call_id}`}
              issueTags={activeIssueTags}
              sttProvider={status?.stt_provider ?? "databricks"}
              languageOptions={status?.languages?.supported}
              defaultLanguage={status?.languages?.default}
              selectedLanguage={interactionLanguage}
              onLanguageChange={onLanguageChange}
              localTurns={conversationByCall[selectedCall.call_id] ?? []}
              onAppendLocalTurn={(turn) => appendTurn(selectedCall.call_id, turn)}
              onUpdateLastCustomerTurn={(turn) => updateLastCustomerTurn(selectedCall.call_id, turn)}
              onRemoveLastCustomerTurn={() => removeLastCustomerTurn(selectedCall.call_id)}
              onResetLocalTurns={() => resetTurns(selectedCall.call_id)}
            />
          ) : (
            <div className="sentient-h-session">
              <SentientHCol step={2} title={copy.onCallTitle} description={copy.noLiveCall}>
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
              <SentientHCol step={3} title={copy.genieColTitle} description="—">
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
              <SentientHCol step={4} title={copy.resolutionColTitle} description="—">
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
