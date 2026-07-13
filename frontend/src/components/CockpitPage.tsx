import { useEffect, useMemo, useState } from "react";
import {
  CustomerWithIssue,
  InteractionLanguage,
  INTERACTION_LANGUAGES,
  StatusResponse,
} from "../api/client";
import architectureHero from "../assets/genie-voice-architecture.jpg";
import { sortCustomersForQueue, SPOTLIGHT_CUSTOMER_ID, useCustomerCall } from "../hooks/useCustomerCall";
import { customerIssueTags } from "../lib/customerIssues";
import { uiCopy } from "../i18n";
import { CockpitSession } from "./CallList";
import {
  letterAt,
  SentientBrandLockup,
  SentientChoice,
  SentientHCol,
  SentientLanguagePicker,
  SentientSessionHead,
  SentientStep,
} from "./sentient/Sentient";

function CockpitLoadingHero({ alt, status }: { alt: string; status: string }) {
  return (
    <div className="sentient-cockpit sentient-cockpit-loading">
      <div className="sentient-arch-hero">
        <SentientBrandLockup className="sentient-brand-lockup-hero" />
        <img src={architectureHero} alt={alt} className="sentient-arch-hero-img" />
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

  const languageOptions =
    status?.languages?.supported && status.languages.supported.length > 0
      ? status.languages.supported
      : INTERACTION_LANGUAGES;

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

  if (customersLoading && !customers.length) {
    return <CockpitLoadingHero alt={copy.cockpitArchitectureAlt} status={copy.connectingWorkspace} />;
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
          <SentientSessionHead
            kicker={copy.activeCustomer}
            name={selectedCustomer?.full_name ?? "…"}
            callLabel={selectedCall ? `${copy.call} ${selectedCall.call_id}` : undefined}
            issues={activeIssueTags}
          />

          <SentientLanguagePicker
            kicker={copy.multilingualUSP}
            description={copy.multilingualUSPDesc}
            options={languageOptions}
            value={interactionLanguage}
            onChange={onLanguageChange}
          />
        </header>

        <div className="sentient-h-rail-cols">
          <SentientHCol
            step={1}
            title="Who needs assist?"
            description="Pick a customer"
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
              sttProvider={status?.stt_provider ?? "deepgram"}
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
              <SentientHCol step={2} title="On the call" description="No live call">
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
              <SentientHCol step={3} title="Genie assist" description="—">
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
              <SentientHCol step={4} title="Resolution" description="—">
                <p className="sentient-muted-text">{copy.noLiveCallDetail}</p>
              </SentientHCol>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
