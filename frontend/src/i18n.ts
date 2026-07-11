import { InteractionLanguage } from "./api/client";

type UiCopy = {
  navCockpit: string;
  navBenchmark: string;
  runtime: string;
  stt: string;
  customersWithIssues: string;
  heroEyebrow: string;
  heroTitle: string;
  heroDescription: string;
  flowVoice: string;
  flowGenie: string;
  flowResolution: string;
  loadingCustomers: string;
  unableToLoadCustomers: string;
  noCustomers: string;
  sidebarTitle: string;
  sidebarSubtitle: string;
  call: string;
  noLiveCall: string;
  noLiveCallDetail: string;
  activeCustomer: string;
  customerProfileLoading: string;
  monthsTenure: (months: number) => string;
  interactionLanguage: string;
  issue: string;
  risk: string;
  recommendedNextAction: string;
  issueResolvedTitle: string;
  issueResolvedDetail: string;
  listeningTitle: string;
  listeningDetail: string;
  conversationStream: string;
  resetScenario: string;
  resetting: string;
  noTranscript: string;
  customer: string;
  agentGenieAssisted: string;
  customerSpeaking: string;
  listening: string;
  transcribingMessage: string;
  preparingGenieResponse: string;
  geniePreparing: string;
  databricksGenieLive: string;
  genieBrandNote: string;
  openInvoices: string;
  overdue: string;
  autopay: string;
  declinedPays: string;
  invoice: string;
  period: string;
  amount: string;
  lateFee: string;
  status: string;
  on: string;
  off: string;
  perMonth: string;
  unavailable: string;
  loading: string;
  resolutionTimeline: string;
  noResolutionEvents: string;
  askGenieLabel: string;
  analyzing: string;
  runGenieQuery: string;
  refreshAssist: string;
  designedAround: (name: string) => string;
  spotlightCustomers: string;
  suggestedAssistQuestion: (customerId: string, callId: string) => string;
  suggestedCallQuestion: (callId: string) => string;
  hideQuery: string;
  showQuery: string;
  issueResolutionJourney: string;
  awaitingCustomer: string;
  awaitingCustomerDetail: string;
  stageDescribe: string;
  stageUnderstand: string;
  stageReview: string;
  stageOffer: string;
  stageApply: string;
  stageClose: string;
  describeInProgress: string;
  capturingSpeech: string;
  understandingRequest: string;
  reviewingFacts: string;
  applyingAgreement: string;
  customerExplains: string;
  customerAskedFor: (items: string) => string;
  paymentPlan: string;
  lateFeeRelief: string;
  genieConfirmedOverdue: (count: number, amount: string) => string;
  accountFactsChecked: string;
  billingConcernIdentified: string;
  accountContextReviewed: string;
  proposedPlanAndWaiver: string;
  proposedLateFeeRelief: string;
  proposedPaymentPlan: string;
  nextStepsShared: string;
  waitingForConfirmation: string;
  issueClosed: string;
  billingUpdated: (invoice: string) => string;
  billingNotUpdated: (reason: string) => string;
  paymentArrangementRecorded: string;
  resolution: string;
  closeBlocked: string;
  billing: string;
  applied: string;
  notApplied: string;
  unknown: string;
  genieValidation: string;
  replyValidated: string;
  replyUnavailable: string;
  speakerCustomer: string;
  speakerAgent: string;
  utterancePlaceholder: string;
  send: string;
  mic: string;
  stopMic: string;
  processingDatabricks: string;
  processingDeepgram: string;
  noDatabricksTranscript: string;
  noDeepgramTranscript: string;
  micAccessError: string;
  micTranscriptionFailed: string;
  captionAvailable: string;
  captionUnavailable: string;
  captionAvailableTitle: string;
  captionUnavailableTitle: string;
  askGenie: string;
  asking: string;
  language: string;
};

const EN: UiCopy = {
  navCockpit: "cockpit",
  navBenchmark: "ASR benchmark",
  runtime: "runtime",
  stt: "stt",
  customersWithIssues: "customers with issues",
  heroEyebrow: "Databricks Genie Voice Agent",
  heroTitle: "Genie-Powered Voice Agent Experience",
  heroDescription:
    "Voice conversations are transcribed by the configured STT provider and enriched with Databricks Genie over governed customer and billing context so agents can resolve calls faster.",
  flowVoice: "Voice Input",
  flowGenie: "Genie Reasoning",
  flowResolution: "Agent Resolution",
  loadingCustomers: "Loading customers with issues...",
  unableToLoadCustomers: "Unable to load customers",
  noCustomers: "No customers with open issues found in account data.",
  sidebarTitle: "Customers with issues",
  sidebarSubtitle: "Billing risk, overdue exposure, and accounts needing agent assist",
  call: "Call",
  noLiveCall: "No live call",
  noLiveCallDetail: "This customer has an open account issue but is not in the active assist queue yet.",
  activeCustomer: "Active Customer",
  customerProfileLoading: "Customer profile loading",
  monthsTenure: (months) => `${months} mo tenure`,
  interactionLanguage: "Interaction language",
  issue: "issue",
  risk: "risk",
  recommendedNextAction: "Recommended next action",
  issueResolvedTitle: "Issue resolved - confirm and close warmly",
  issueResolvedDetail:
    "Payment arrangement and waiver are applied. Confirm closure with the customer and offer brief follow-up help.",
  listeningTitle: "Listening to customer context",
  listeningDetail:
    "Collecting customer request first. Recommended next action will appear right after the Genie-assisted agent response.",
  conversationStream: "Conversation stream (voice to Genie to agent)",
  resetScenario: "Reset scenario",
  resetting: "Resetting...",
  noTranscript: "No transcript captured.",
  customer: "Customer",
  agentGenieAssisted: "Agent (Genie-assisted)",
  customerSpeaking: "Customer (speaking)",
  listening: "Listening...",
  transcribingMessage: "Transcribing your message...",
  preparingGenieResponse: "Preparing Genie-assisted response...",
  geniePreparing: "Genie is preparing the agent response...",
  databricksGenieLive: "Databricks Genie live intelligence",
  genieBrandNote:
    "Genie reads governed customer, invoice, payment, and call context to guide voice agents.",
  openInvoices: "Open invoices",
  overdue: "Overdue",
  autopay: "Autopay",
  declinedPays: "Declined pays",
  invoice: "Invoice",
  period: "Period",
  amount: "Amount",
  lateFee: "Late fee",
  status: "Status",
  on: "on",
  off: "off",
  perMonth: "mo",
  unavailable: "unavailable",
  loading: "loading...",
  resolutionTimeline: "Resolution timeline",
  noResolutionEvents: "No resolution events yet.",
  askGenieLabel: "Ask Genie for a real-time assist prompt",
  analyzing: "Analyzing...",
  runGenieQuery: "Run Genie Query",
  refreshAssist: "Refresh Assist",
  designedAround: (name) =>
    `Designed around ${name}: ask for payment arrangement + late fee relief.`,
  spotlightCustomers: "spotlight customers",
  suggestedAssistQuestion: (customerId, callId) =>
    `For customer ${customerId} on call ${callId}, summarize account risk, overdue/declined payment context, and provide the best retention-safe next action for the agent.`,
  suggestedCallQuestion: (callId) =>
    `Give a live assist summary for call ${callId}, including likely intent and next best action.`,
  hideQuery: "Hide query",
  showQuery: "Show query",
  issueResolutionJourney: "Issue resolution journey",
  awaitingCustomer: "Awaiting customer",
  awaitingCustomerDetail: "The resolution journey begins when the customer describes their issue.",
  stageDescribe: "Customer describes the issue",
  stageUnderstand: "Request understood",
  stageReview: "Account reviewed with Genie",
  stageOffer: "Resolution offered to customer",
  stageApply: "Agreement applied to billing",
  stageClose: "Issue closed",
  describeInProgress: "Listening - customer is explaining the issue...",
  capturingSpeech: "Capturing what the customer said...",
  understandingRequest: "Understanding the customer's billing request...",
  reviewingFacts: "Genie is reviewing account facts and preparing the resolution offer for the agent...",
  applyingAgreement: "Applying the agreed payment arrangement and waiver to billing...",
  customerExplains: "Customer explains their billing concern on the call.",
  customerAskedFor: (items) => `customer asked for ${items}`,
  paymentPlan: "payment plan",
  lateFeeRelief: "late fee relief",
  genieConfirmedOverdue: (count, amount) =>
    `Genie confirmed ${count} overdue invoice(s) totaling ${amount}.`,
  accountFactsChecked: "Account facts checked against governed billing records.",
  billingConcernIdentified: "Billing concern identified from the conversation.",
  accountContextReviewed: "Account context reviewed before offering next steps.",
  proposedPlanAndWaiver: "Agent proposed a payment arrangement and late fee waiver.",
  proposedLateFeeRelief: "Agent proposed late fee relief on the overdue balance.",
  proposedPaymentPlan: "Agent proposed a payment arrangement.",
  nextStepsShared: "Agent shared next steps to resolve the billing issue.",
  waitingForConfirmation: "Waiting for customer confirmation before updating billing.",
  issueClosed: "Issue closed - customer informed that changes will appear on the next statement.",
  billingUpdated: (invoice) => `Billing updated (${invoice}).`,
  billingNotUpdated: (reason) => `Billing not updated: ${reason}.`,
  paymentArrangementRecorded: "Payment arrangement and waiver recorded on the account.",
  resolution: "Resolution",
  closeBlocked: "Close blocked",
  billing: "Billing",
  applied: "applied",
  notApplied: "not applied",
  unknown: "unknown",
  genieValidation: "Genie validation",
  replyValidated: "reply validated",
  replyUnavailable: "reply unavailable",
  speakerCustomer: "Customer",
  speakerAgent: "Agent",
  utterancePlaceholder: "Type the next customer/agent utterance...",
  send: "Send",
  mic: "Mic",
  stopMic: "Stop Mic",
  processingDatabricks: "Processing voice with Databricks model...",
  processingDeepgram: "Processing voice with Deepgram...",
  noDatabricksTranscript: "No transcript returned from Databricks model",
  noDeepgramTranscript: "No transcript returned from Deepgram",
  micAccessError: "Unable to access microphone",
  micTranscriptionFailed: "mic transcription failed",
  captionAvailable: "Live caption available - final transcript by Databricks model",
  captionUnavailable: "Live caption unavailable in this browser - Databricks transcript on stop",
  captionAvailableTitle:
    "Your browser supports a live on-screen caption while you speak. The Databricks model still produces the final transcript on stop.",
  captionUnavailableTitle:
    "This browser has no Web Speech API, so the live caption is skipped. Recording and the Databricks transcript are unaffected. Try Chrome, Edge, or Safari.",
  askGenie: "Ask Genie",
  asking: "Asking...",
  language: "Language",
};

const TH: UiCopy = {
  ...EN,
  navCockpit: "ค็อกพิท",
  navBenchmark: "เบนช์มาร์ก ASR",
  runtime: "รันไทม์",
  stt: "STT",
  customersWithIssues: "ลูกค้าที่มีปัญหา",
  heroEyebrow: "Databricks Genie Voice Agent",
  heroTitle: "ประสบการณ์ผู้ช่วยเสียงที่ขับเคลื่อนด้วย Genie",
  heroDescription:
    "ถอดเสียงบทสนทนาด้วยผู้ให้บริการ STT ที่กำหนดไว้ และเสริมด้วย Databricks Genie บนข้อมูลลูกค้าและการเรียกเก็บเงินที่กำกับดูแลแล้ว เพื่อช่วยให้เจ้าหน้าที่ปิดเคสได้เร็วขึ้น",
  flowVoice: "เสียงเข้า",
  flowGenie: "Genie วิเคราะห์",
  flowResolution: "เจ้าหน้าที่แก้ไขปัญหา",
  loadingCustomers: "กำลังโหลดลูกค้าที่มีปัญหา...",
  unableToLoadCustomers: "โหลดรายชื่อลูกค้าไม่ได้",
  noCustomers: "ไม่พบลูกค้าที่มีปัญหาเปิดอยู่ในข้อมูลบัญชี",
  sidebarTitle: "ลูกค้าที่มีปัญหา",
  sidebarSubtitle: "ความเสี่ยงการเรียกเก็บเงิน ยอดค้างชำระ และบัญชีที่ต้องการความช่วยเหลือ",
  call: "สาย",
  noLiveCall: "ไม่มีสายสด",
  noLiveCallDetail: "ลูกค้ารายนี้มีปัญหาบัญชีเปิดอยู่ แต่ยังไม่ได้อยู่ในคิวช่วยเหลือสด",
  activeCustomer: "ลูกค้าที่กำลังดู",
  customerProfileLoading: "กำลังโหลดโปรไฟล์ลูกค้า",
  monthsTenure: (months) => `ใช้งานมา ${months} เดือน`,
  interactionLanguage: "ภาษาที่ใช้โต้ตอบ",
  issue: "ปัญหา",
  risk: "ความเสี่ยง",
  recommendedNextAction: "คำแนะนำถัดไป",
  issueResolvedTitle: "แก้ไขปัญหาแล้ว - ยืนยันและปิดเคสอย่างสุภาพ",
  issueResolvedDetail:
    "มีการตั้งแผนชำระเงินและยกเว้นค่าธรรมเนียมแล้ว ให้ยืนยันการปิดเคสกับลูกค้าและเสนอความช่วยเหลือเพิ่มเติมสั้น ๆ",
  listeningTitle: "กำลังฟังบริบทจากลูกค้า",
  listeningDetail:
    "กำลังรวบรวมคำขอจากลูกค้าก่อน คำแนะนำถัดไปจะแสดงหลังจาก Genie เตรียมคำตอบให้เจ้าหน้าที่",
  conversationStream: "สตรีมบทสนทนา (เสียงสู่ Genie สู่เจ้าหน้าที่)",
  resetScenario: "รีเซ็ตสถานการณ์",
  resetting: "กำลังรีเซ็ต...",
  noTranscript: "ยังไม่มีข้อความถอดเสียง",
  customer: "ลูกค้า",
  agentGenieAssisted: "เจ้าหน้าที่ (ช่วยโดย Genie)",
  customerSpeaking: "ลูกค้า (กำลังพูด)",
  listening: "กำลังฟัง...",
  transcribingMessage: "กำลังถอดเสียงข้อความของคุณ...",
  preparingGenieResponse: "กำลังเตรียมคำตอบจาก Genie...",
  geniePreparing: "Genie กำลังเตรียมคำตอบให้เจ้าหน้าที่...",
  databricksGenieLive: "ข้อมูลอัจฉริยะสดจาก Databricks Genie",
  genieBrandNote:
    "Genie อ่านบริบทลูกค้า ใบแจ้งหนี้ การชำระเงิน และสายสนทนาที่กำกับดูแลแล้ว เพื่อช่วยเจ้าหน้าที่เสียง",
  openInvoices: "ใบแจ้งหนี้เปิด",
  overdue: "ค้างชำระ",
  autopay: "ชำระอัตโนมัติ",
  declinedPays: "ชำระเงินไม่สำเร็จ",
  invoice: "ใบแจ้งหนี้",
  period: "รอบบิล",
  amount: "จำนวนเงิน",
  lateFee: "ค่าปรับล่าช้า",
  status: "สถานะ",
  on: "เปิด",
  off: "ปิด",
  perMonth: "เดือน",
  unavailable: "ไม่พร้อมใช้งาน",
  loading: "กำลังโหลด...",
  resolutionTimeline: "ไทม์ไลน์การแก้ไข",
  noResolutionEvents: "ยังไม่มีเหตุการณ์การแก้ไข",
  askGenieLabel: "ถาม Genie เพื่อขอคำแนะนำแบบเรียลไทม์",
  analyzing: "กำลังวิเคราะห์...",
  runGenieQuery: "ถาม Genie",
  refreshAssist: "รีเฟรชคำแนะนำ",
  designedAround: (name) => `ออกแบบสำหรับ ${name}: ถามเรื่องแผนชำระเงินและการยกเว้นค่าปรับ`,
  spotlightCustomers: "ลูกค้าสำคัญ",
  suggestedAssistQuestion: (customerId, callId) =>
    `สำหรับลูกค้า ${customerId} ในสาย ${callId} ช่วยสรุปความเสี่ยงของบัญชี บริบทใบแจ้งหนี้ค้างชำระหรือการชำระเงินไม่สำเร็จ และเสนอขั้นตอนถัดไปที่เหมาะสมที่สุดให้เจ้าหน้าที่ โดยยังยึดข้อมูลบัญชีจริง`,
  suggestedCallQuestion: (callId) =>
    `ช่วยสรุปคำแนะนำสำหรับสาย ${callId} รวมถึงเจตนาที่เป็นไปได้และขั้นตอนถัดไปที่เหมาะสม`,
  hideQuery: "ซ่อนคำสั่ง SQL",
  showQuery: "แสดงคำสั่ง SQL",
  issueResolutionJourney: "เส้นทางการแก้ไขปัญหา",
  awaitingCustomer: "รอลูกค้า",
  awaitingCustomerDetail: "เส้นทางการแก้ไขจะเริ่มเมื่อลูกค้าอธิบายปัญหา",
  stageDescribe: "ลูกค้าอธิบายปัญหา",
  stageUnderstand: "เข้าใจคำขอแล้ว",
  stageReview: "ตรวจสอบบัญชีด้วย Genie",
  stageOffer: "เสนอแนวทางแก้ไขให้ลูกค้า",
  stageApply: "บันทึกข้อตกลงในระบบบิล",
  stageClose: "ปิดเคส",
  describeInProgress: "กำลังฟัง - ลูกค้ากำลังอธิบายปัญหา...",
  capturingSpeech: "กำลังบันทึกสิ่งที่ลูกค้าพูด...",
  understandingRequest: "กำลังทำความเข้าใจคำขอด้านบิลของลูกค้า...",
  reviewingFacts: "Genie กำลังตรวจสอบข้อมูลบัญชีและเตรียมข้อเสนอการแก้ไขให้เจ้าหน้าที่...",
  applyingAgreement: "กำลังบันทึกแผนชำระเงินและการยกเว้นค่าปรับที่ตกลงกัน...",
  customerExplains: "ลูกค้าอธิบายปัญหาด้านบิลในสายสนทนา",
  customerAskedFor: (items) => `ลูกค้าขอ ${items}`,
  paymentPlan: "แผนชำระเงิน",
  lateFeeRelief: "การยกเว้นค่าปรับ",
  genieConfirmedOverdue: (count, amount) =>
    `Genie ยืนยันว่ามีใบแจ้งหนี้ค้างชำระ ${count} ใบ รวม ${amount}`,
  accountFactsChecked: "ตรวจสอบข้อมูลบัญชีกับระเบียนบิลที่กำกับดูแลแล้ว",
  billingConcernIdentified: "ระบุปัญหาด้านบิลจากบทสนทนาแล้ว",
  accountContextReviewed: "ตรวจสอบบริบทบัญชีก่อนเสนอขั้นตอนถัดไปแล้ว",
  proposedPlanAndWaiver: "เจ้าหน้าที่เสนอแผนชำระเงินและการยกเว้นค่าปรับ",
  proposedLateFeeRelief: "เจ้าหน้าที่เสนอการยกเว้นค่าปรับสำหรับยอดค้างชำระ",
  proposedPaymentPlan: "เจ้าหน้าที่เสนอแผนชำระเงิน",
  nextStepsShared: "เจ้าหน้าที่แจ้งขั้นตอนถัดไปเพื่อแก้ไขปัญหาบิลแล้ว",
  waitingForConfirmation: "รอการยืนยันจากลูกค้าก่อนอัปเดตบิล",
  issueClosed: "ปิดเคสแล้ว - แจ้งลูกค้าว่าการเปลี่ยนแปลงจะแสดงในใบแจ้งยอดถัดไป",
  billingUpdated: (invoice) => `อัปเดตบิลแล้ว (${invoice})`,
  billingNotUpdated: (reason) => `ยังไม่ได้อัปเดตบิล: ${reason}`,
  paymentArrangementRecorded: "บันทึกแผนชำระเงินและการยกเว้นค่าปรับในบัญชีแล้ว",
  resolution: "การแก้ไข",
  closeBlocked: "ยังปิดเคสไม่ได้",
  billing: "บิล",
  applied: "บันทึกแล้ว",
  notApplied: "ยังไม่บันทึก",
  unknown: "ไม่ทราบ",
  genieValidation: "การตรวจสอบโดย Genie",
  replyValidated: "ตรวจสอบคำตอบแล้ว",
  replyUnavailable: "ไม่มีคำตอบ",
  speakerCustomer: "ลูกค้า",
  speakerAgent: "เจ้าหน้าที่",
  utterancePlaceholder: "พิมพ์ข้อความถัดไปของลูกค้าหรือเจ้าหน้าที่...",
  send: "ส่ง",
  mic: "ไมค์",
  stopMic: "หยุดไมค์",
  processingDatabricks: "กำลังประมวลผลเสียงด้วยโมเดล Databricks...",
  processingDeepgram: "กำลังประมวลผลเสียงด้วย Deepgram...",
  noDatabricksTranscript: "ไม่มีข้อความถอดเสียงจากโมเดล Databricks",
  noDeepgramTranscript: "ไม่มีข้อความถอดเสียงจาก Deepgram",
  micAccessError: "เข้าถึงไมโครโฟนไม่ได้",
  micTranscriptionFailed: "ถอดเสียงจากไมค์ไม่สำเร็จ",
  captionAvailable: "มีคำบรรยายสด - ข้อความสุดท้ายมาจากโมเดล Databricks",
  captionUnavailable: "เบราว์เซอร์นี้ไม่มีคำบรรยายสด - ถอดเสียงด้วย Databricks เมื่อหยุดพูด",
  captionAvailableTitle:
    "เบราว์เซอร์รองรับคำบรรยายบนหน้าจอระหว่างพูด โดยโมเดล Databricks ยังเป็นผู้สร้างข้อความถอดเสียงสุดท้าย",
  captionUnavailableTitle:
    "เบราว์เซอร์นี้ไม่มี Web Speech API จึงข้ามคำบรรยายสด แต่การบันทึกและการถอดเสียงด้วย Databricks ยังทำงานได้",
  askGenie: "ถาม Genie",
  asking: "กำลังถาม...",
  language: "ภาษา",
};

const ID: UiCopy = {
  ...EN,
  navCockpit: "kokpit",
  navBenchmark: "benchmark ASR",
  runtime: "runtime",
  stt: "STT",
  customersWithIssues: "pelanggan bermasalah",
  heroEyebrow: "Agen Suara Databricks Genie",
  heroTitle: "Pengalaman Agen Suara Berbasis Genie",
  heroDescription:
    "Percakapan suara ditranskripsi oleh penyedia STT yang dikonfigurasi dan diperkaya dengan Databricks Genie di atas konteks pelanggan dan penagihan yang terkelola agar agen dapat menyelesaikan panggilan lebih cepat.",
  flowVoice: "Input Suara",
  flowGenie: "Penalaran Genie",
  flowResolution: "Resolusi Agen",
  loadingCustomers: "Memuat pelanggan bermasalah...",
  unableToLoadCustomers: "Tidak dapat memuat pelanggan",
  noCustomers: "Tidak ada pelanggan dengan masalah terbuka di data akun.",
  interactionLanguage: "Bahasa interaksi",
  sidebarTitle: "Pelanggan bermasalah",
  sidebarSubtitle: "Risiko penagihan, eksposur jatuh tempo, dan akun yang perlu bantuan agen",
  call: "Panggilan",
  noLiveCall: "Tidak ada panggilan live",
  noLiveCallDetail: "Pelanggan ini memiliki masalah akun terbuka tetapi belum ada di antrean bantuan aktif.",
  activeCustomer: "Pelanggan aktif",
  customerProfileLoading: "Memuat profil pelanggan",
  monthsTenure: (months) => `${months} bulan berlangganan`,
  issue: "masalah",
  risk: "risiko",
  recommendedNextAction: "Tindakan berikutnya yang disarankan",
  issueResolvedTitle: "Masalah selesai - konfirmasi dan tutup dengan hangat",
  issueResolvedDetail:
    "Pengaturan pembayaran dan penghapusan biaya sudah diterapkan. Konfirmasi penutupan dengan pelanggan dan tawarkan bantuan singkat jika perlu.",
  listeningTitle: "Mendengarkan konteks pelanggan",
  listeningDetail:
    "Mengumpulkan permintaan pelanggan terlebih dahulu. Tindakan berikutnya akan muncul setelah respons agen berbantuan Genie siap.",
  conversationStream: "Alur percakapan (suara ke Genie ke agen)",
  resetScenario: "Reset skenario",
  resetting: "Mereset...",
  noTranscript: "Belum ada transkrip.",
  customer: "Pelanggan",
  agentGenieAssisted: "Agen (dibantu Genie)",
  customerSpeaking: "Pelanggan (berbicara)",
  listening: "Mendengarkan...",
  transcribingMessage: "Mentranskripsi pesan Anda...",
  preparingGenieResponse: "Menyiapkan respons berbantuan Genie...",
  geniePreparing: "Genie sedang menyiapkan respons untuk agen...",
  databricksGenieLive: "Intelijen live Databricks Genie",
  genieBrandNote:
    "Genie membaca konteks pelanggan, invoice, pembayaran, dan panggilan yang terkelola untuk memandu agen suara.",
  openInvoices: "Invoice terbuka",
  overdue: "Jatuh tempo",
  autopay: "Autopay",
  declinedPays: "Pembayaran ditolak",
  invoice: "Invoice",
  period: "Periode",
  amount: "Jumlah",
  lateFee: "Biaya keterlambatan",
  status: "Status",
  on: "aktif",
  off: "nonaktif",
  perMonth: "bln",
  unavailable: "tidak tersedia",
  loading: "memuat...",
  resolutionTimeline: "Linimasa penyelesaian",
  noResolutionEvents: "Belum ada peristiwa penyelesaian.",
  askGenieLabel: "Tanya Genie untuk prompt bantuan real-time",
  analyzing: "Menganalisis...",
  runGenieQuery: "Jalankan Genie",
  refreshAssist: "Refresh Bantuan",
  designedAround: (name) => `Dirancang untuk ${name}: tanyakan pengaturan pembayaran + keringanan biaya keterlambatan.`,
  spotlightCustomers: "pelanggan prioritas",
  suggestedAssistQuestion: (customerId, callId) =>
    `Untuk pelanggan ${customerId} pada panggilan ${callId}, ringkas risiko akun, konteks invoice jatuh tempo/pembayaran ditolak, dan berikan tindakan berikutnya yang paling aman untuk retensi bagi agen.`,
  suggestedCallQuestion: (callId) =>
    `Berikan ringkasan bantuan live untuk panggilan ${callId}, termasuk kemungkinan intent dan tindakan berikutnya.`,
  hideQuery: "Sembunyikan query",
  showQuery: "Tampilkan query",
  issueResolutionJourney: "Perjalanan penyelesaian masalah",
  awaitingCustomer: "Menunggu pelanggan",
  awaitingCustomerDetail: "Perjalanan penyelesaian dimulai saat pelanggan menjelaskan masalahnya.",
  stageDescribe: "Pelanggan menjelaskan masalah",
  stageUnderstand: "Permintaan dipahami",
  stageReview: "Akun ditinjau dengan Genie",
  stageOffer: "Resolusi ditawarkan ke pelanggan",
  stageApply: "Kesepakatan diterapkan ke penagihan",
  stageClose: "Masalah ditutup",
  describeInProgress: "Mendengarkan - pelanggan sedang menjelaskan masalah...",
  capturingSpeech: "Menangkap ucapan pelanggan...",
  understandingRequest: "Memahami permintaan penagihan pelanggan...",
  reviewingFacts: "Genie meninjau fakta akun dan menyiapkan penawaran resolusi untuk agen...",
  applyingAgreement: "Menerapkan pengaturan pembayaran dan penghapusan biaya yang disepakati...",
  customerExplains: "Pelanggan menjelaskan masalah penagihan dalam panggilan.",
  customerAskedFor: (items) => `pelanggan meminta ${items}`,
  paymentPlan: "pengaturan pembayaran",
  lateFeeRelief: "keringanan biaya keterlambatan",
  genieConfirmedOverdue: (count, amount) =>
    `Genie mengonfirmasi ${count} invoice jatuh tempo dengan total ${amount}.`,
  accountFactsChecked: "Fakta akun diperiksa terhadap catatan penagihan terkelola.",
  billingConcernIdentified: "Masalah penagihan teridentifikasi dari percakapan.",
  accountContextReviewed: "Konteks akun ditinjau sebelum menawarkan langkah berikutnya.",
  proposedPlanAndWaiver: "Agen menawarkan pengaturan pembayaran dan penghapusan biaya keterlambatan.",
  proposedLateFeeRelief: "Agen menawarkan keringanan biaya keterlambatan pada saldo jatuh tempo.",
  proposedPaymentPlan: "Agen menawarkan pengaturan pembayaran.",
  nextStepsShared: "Agen membagikan langkah berikutnya untuk menyelesaikan masalah penagihan.",
  waitingForConfirmation: "Menunggu konfirmasi pelanggan sebelum memperbarui penagihan.",
  issueClosed: "Masalah ditutup - pelanggan diberi tahu perubahan akan muncul pada tagihan berikutnya.",
  billingUpdated: (invoice) => `Penagihan diperbarui (${invoice}).`,
  billingNotUpdated: (reason) => `Penagihan belum diperbarui: ${reason}.`,
  paymentArrangementRecorded: "Pengaturan pembayaran dan penghapusan biaya dicatat pada akun.",
  resolution: "Resolusi",
  closeBlocked: "Penutupan tertahan",
  billing: "Penagihan",
  applied: "diterapkan",
  notApplied: "belum diterapkan",
  unknown: "tidak diketahui",
  genieValidation: "Validasi Genie",
  replyValidated: "respons tervalidasi",
  replyUnavailable: "respons tidak tersedia",
  speakerCustomer: "Pelanggan",
  speakerAgent: "Agen",
  utterancePlaceholder: "Ketik ujaran pelanggan/agen berikutnya...",
  send: "Kirim",
  mic: "Mik",
  stopMic: "Stop Mik",
  processingDatabricks: "Memproses suara dengan model Databricks...",
  processingDeepgram: "Memproses suara dengan Deepgram...",
  noDatabricksTranscript: "Tidak ada transkrip dari model Databricks",
  noDeepgramTranscript: "Tidak ada transkrip dari Deepgram",
  micAccessError: "Tidak dapat mengakses mikrofon",
  micTranscriptionFailed: "transkripsi mikrofon gagal",
  captionAvailable: "Caption live tersedia - transkrip final oleh model Databricks",
  captionUnavailable: "Caption live tidak tersedia di browser ini - transkrip Databricks saat berhenti",
  captionAvailableTitle:
    "Browser mendukung caption di layar saat Anda berbicara. Model Databricks tetap menghasilkan transkrip final saat berhenti.",
  captionUnavailableTitle:
    "Browser ini tidak memiliki Web Speech API, jadi caption live dilewati. Rekaman dan transkrip Databricks tetap berjalan.",
  askGenie: "Tanya Genie",
  asking: "Bertanya...",
  language: "Bahasa",
};

const ZH: UiCopy = {
  ...EN,
  navCockpit: "工作台",
  navBenchmark: "ASR 基准",
  runtime: "运行模式",
  stt: "STT",
  customersWithIssues: "有问题的客户",
  heroEyebrow: "Databricks Genie 语音客服",
  heroTitle: "Genie 驱动的语音客服体验",
  heroDescription:
    "语音对话由配置的 STT 服务转写，并通过 Databricks Genie 结合受治理的客户和账单上下文进行增强，帮助客服更快解决来电。",
  flowVoice: "语音输入",
  flowGenie: "Genie 推理",
  flowResolution: "客服解决",
  loadingCustomers: "正在加载有问题的客户...",
  unableToLoadCustomers: "无法加载客户",
  noCustomers: "账户数据中没有发现未关闭的问题客户。",
  interactionLanguage: "交互语言",
  sidebarTitle: "有问题的客户",
  sidebarSubtitle: "账单风险、逾期风险，以及需要客服协助的账户",
  call: "来电",
  noLiveCall: "没有实时来电",
  noLiveCallDetail: "该客户有未关闭的账户问题，但尚未进入实时协助队列。",
  activeCustomer: "当前客户",
  customerProfileLoading: "正在加载客户资料",
  monthsTenure: (months) => `${months} 个月在网`,
  issue: "问题",
  risk: "风险",
  recommendedNextAction: "建议的下一步操作",
  issueResolvedTitle: "问题已解决 - 确认并友好结束",
  issueResolvedDetail:
    "付款安排和费用减免已应用。请向客户确认关闭，并简短提供后续帮助。",
  listeningTitle: "正在听取客户上下文",
  listeningDetail:
    "先收集客户诉求。Genie 辅助的客服回复完成后，将显示建议的下一步操作。",
  conversationStream: "对话流（语音到 Genie 到客服）",
  resetScenario: "重置场景",
  resetting: "正在重置...",
  noTranscript: "尚无转写内容。",
  customer: "客户",
  agentGenieAssisted: "客服（Genie 辅助）",
  customerSpeaking: "客户（正在说话）",
  listening: "正在听...",
  transcribingMessage: "正在转写您的消息...",
  preparingGenieResponse: "正在准备 Genie 辅助回复...",
  geniePreparing: "Genie 正在为客服准备回复...",
  databricksGenieLive: "Databricks Genie 实时智能",
  genieBrandNote:
    "Genie 读取受治理的客户、发票、付款和通话上下文，辅助语音客服。",
  openInvoices: "未结发票",
  overdue: "逾期",
  autopay: "自动付款",
  declinedPays: "付款失败",
  invoice: "发票",
  period: "账期",
  amount: "金额",
  lateFee: "滞纳金",
  status: "状态",
  on: "开启",
  off: "关闭",
  perMonth: "月",
  unavailable: "不可用",
  loading: "加载中...",
  resolutionTimeline: "解决时间线",
  noResolutionEvents: "暂无解决事件。",
  askGenieLabel: "向 Genie 请求实时协助提示",
  analyzing: "正在分析...",
  runGenieQuery: "运行 Genie 查询",
  refreshAssist: "刷新辅助",
  designedAround: (name) => `围绕 ${name} 设计：询问付款安排和滞纳金减免。`,
  spotlightCustomers: "重点客户",
  suggestedAssistQuestion: (customerId, callId) =>
    `对于客户 ${customerId} 在来电 ${callId} 中，请总结账户风险、逾期/付款失败上下文，并给出最适合客服的留存安全下一步操作。`,
  suggestedCallQuestion: (callId) =>
    `请为来电 ${callId} 生成实时协助摘要，包括可能意图和下一步操作。`,
  hideQuery: "隐藏查询",
  showQuery: "显示查询",
  issueResolutionJourney: "问题解决路径",
  awaitingCustomer: "等待客户",
  awaitingCustomerDetail: "客户描述问题后，解决路径开始。",
  stageDescribe: "客户描述问题",
  stageUnderstand: "已理解请求",
  stageReview: "用 Genie 查看账户",
  stageOffer: "向客户提供解决方案",
  stageApply: "将协议应用到账单",
  stageClose: "关闭问题",
  describeInProgress: "正在听 - 客户正在说明问题...",
  capturingSpeech: "正在捕获客户所说内容...",
  understandingRequest: "正在理解客户的账单请求...",
  reviewingFacts: "Genie 正在查看账户事实并为客服准备解决方案...",
  applyingAgreement: "正在应用已同意的付款安排和费用减免...",
  customerExplains: "客户在通话中说明账单问题。",
  customerAskedFor: (items) => `客户请求 ${items}`,
  paymentPlan: "付款安排",
  lateFeeRelief: "滞纳金减免",
  genieConfirmedOverdue: (count, amount) =>
    `Genie 确认有 ${count} 张逾期发票，总计 ${amount}。`,
  accountFactsChecked: "账户事实已根据受治理的账单记录检查。",
  billingConcernIdentified: "已从对话中识别账单问题。",
  accountContextReviewed: "提供下一步前已查看账户上下文。",
  proposedPlanAndWaiver: "客服提出付款安排和滞纳金减免。",
  proposedLateFeeRelief: "客服提出对逾期余额减免滞纳金。",
  proposedPaymentPlan: "客服提出付款安排。",
  nextStepsShared: "客服说明了解决账单问题的下一步。",
  waitingForConfirmation: "等待客户确认后再更新账单。",
  issueClosed: "问题已关闭 - 已告知客户变更会显示在下一期账单中。",
  billingUpdated: (invoice) => `账单已更新（${invoice}）。`,
  billingNotUpdated: (reason) => `账单未更新：${reason}。`,
  paymentArrangementRecorded: "付款安排和费用减免已记录到账户。",
  resolution: "解决方案",
  closeBlocked: "暂不能关闭",
  billing: "账单",
  applied: "已应用",
  notApplied: "未应用",
  unknown: "未知",
  genieValidation: "Genie 验证",
  replyValidated: "回复已验证",
  replyUnavailable: "回复不可用",
  speakerCustomer: "客户",
  speakerAgent: "客服",
  utterancePlaceholder: "输入下一条客户/客服话语...",
  send: "发送",
  mic: "麦克风",
  stopMic: "停止麦克风",
  processingDatabricks: "正在用 Databricks 模型处理语音...",
  processingDeepgram: "正在用 Deepgram 处理语音...",
  noDatabricksTranscript: "Databricks 模型没有返回转写",
  noDeepgramTranscript: "Deepgram 没有返回转写",
  micAccessError: "无法访问麦克风",
  micTranscriptionFailed: "麦克风转写失败",
  captionAvailable: "可用实时字幕 - 最终转写由 Databricks 模型生成",
  captionUnavailable: "此浏览器没有实时字幕 - 停止后由 Databricks 转写",
  captionAvailableTitle:
    "浏览器支持说话时显示屏幕字幕。Databricks 模型仍会在停止后生成最终转写。",
  captionUnavailableTitle:
    "此浏览器没有 Web Speech API，因此跳过实时字幕。录音和 Databricks 转写不受影响。",
  askGenie: "询问 Genie",
  asking: "正在询问...",
  language: "语言",
};

const COPY: Record<InteractionLanguage, UiCopy> = {
  "en-US": EN,
  "th-TH": TH,
  "id-ID": ID,
  "zh-CN": ZH,
};

export function uiCopy(language: InteractionLanguage | undefined): UiCopy {
  return COPY[language ?? "en-US"] ?? EN;
}

const VALUE_LABELS: Record<InteractionLanguage, Record<string, Record<string, string>>> = {
  "en-US": {
    status: {
      open: "open",
      closed: "closed",
      in_progress: "in progress",
      overdue: "overdue",
      paid: "paid",
      resolved: "resolved",
      active: "active",
      at_risk: "at risk",
      neutral: "neutral",
      negative: "negative",
      positive: "positive",
      stable: "stable",
      elevated: "elevated",
      status_changed: "status changed",
    },
    reason: {
      no_overdue_invoice: "no overdue invoice",
      agent_reply_unavailable: "agent reply unavailable",
      pending_customer_confirmation: "pending customer confirmation",
      reply_failed_metric_validation: "reply failed validation",
      language_mismatch: "language mismatch",
    },
  },
  "th-TH": {
    status: {
      open: "เปิดอยู่",
      closed: "ปิดแล้ว",
      in_progress: "กำลังดำเนินการ",
      overdue: "ค้างชำระ",
      paid: "ชำระแล้ว",
      resolved: "แก้ไขแล้ว",
      active: "ใช้งานอยู่",
      at_risk: "มีความเสี่ยง",
      neutral: "เป็นกลาง",
      negative: "เชิงลบ",
      positive: "เชิงบวก",
      stable: "คงที่",
      elevated: "สูงขึ้น",
      status_changed: "เปลี่ยนสถานะ",
    },
    intent: {
      billing_dispute: "โต้แย้งบิล",
      late_fee: "ค่าปรับล่าช้า",
      payment_arrangement: "จัดแผนชำระเงิน",
      refund: "ขอคืนเงิน",
      autopay_issue: "ปัญหาชำระอัตโนมัติ",
      plan_inquiry: "สอบถามแพ็กเกจ",
      cancellation_risk: "เสี่ยงยกเลิก",
      billing_inquiry: "สอบถามบิล",
    },
    profile: {
      consumer: "ลูกค้าบุคคล",
      business: "ลูกค้าธุรกิจ",
      premium: "พรีเมียม",
      standard: "มาตรฐาน",
      basic: "พื้นฐาน",
      apac: "เอเชียแปซิฟิก",
      emea: "ยุโรป ตะวันออกกลาง และแอฟริกา",
      amer: "อเมริกา",
    },
    reason: {
      no_overdue_invoice: "ไม่พบใบแจ้งหนี้ค้างชำระ",
      agent_reply_unavailable: "ไม่มีคำตอบสำหรับเจ้าหน้าที่",
      pending_customer_confirmation: "รอการยืนยันจากลูกค้า",
      reply_failed_metric_validation: "คำตอบไม่ผ่านการตรวจสอบข้อมูล",
      language_mismatch: "ภาษาไม่ตรงกับที่เลือก",
    },
  },
  "id-ID": {
    status: {
      open: "terbuka",
      closed: "selesai",
      in_progress: "sedang diproses",
      overdue: "jatuh tempo",
      paid: "dibayar",
      resolved: "terselesaikan",
      active: "aktif",
      at_risk: "berisiko",
      neutral: "netral",
      negative: "negatif",
      positive: "positif",
      stable: "stabil",
      elevated: "meningkat",
      status_changed: "status berubah",
    },
    intent: {
      billing_dispute: "sengketa tagihan",
      late_fee: "biaya keterlambatan",
      payment_arrangement: "pengaturan pembayaran",
      refund: "permintaan refund",
      autopay_issue: "masalah autopay",
      plan_inquiry: "pertanyaan paket",
      cancellation_risk: "risiko pembatalan",
      billing_inquiry: "pertanyaan tagihan",
    },
    profile: {
      consumer: "konsumen",
      business: "bisnis",
      premium: "premium",
      standard: "standar",
      basic: "dasar",
      apac: "APAC",
      emea: "EMEA",
      amer: "Amerika",
    },
    reason: {
      no_overdue_invoice: "tidak ada invoice jatuh tempo",
      agent_reply_unavailable: "respons agen tidak tersedia",
      pending_customer_confirmation: "menunggu konfirmasi pelanggan",
      reply_failed_metric_validation: "respons gagal validasi metrik",
      language_mismatch: "bahasa tidak sesuai",
    },
  },
  "zh-CN": {
    status: {
      open: "未关闭",
      closed: "已关闭",
      in_progress: "处理中",
      overdue: "逾期",
      paid: "已支付",
      resolved: "已解决",
      active: "正常",
      at_risk: "有风险",
      neutral: "中性",
      negative: "负面",
      positive: "正面",
      stable: "稳定",
      elevated: "升高",
      status_changed: "状态已变更",
    },
    intent: {
      billing_dispute: "账单争议",
      late_fee: "滞纳金",
      payment_arrangement: "付款安排",
      refund: "退款请求",
      autopay_issue: "自动付款问题",
      plan_inquiry: "套餐咨询",
      cancellation_risk: "取消风险",
      billing_inquiry: "账单咨询",
    },
    profile: {
      consumer: "个人客户",
      business: "企业客户",
      premium: "高级",
      standard: "标准",
      basic: "基础",
      apac: "亚太",
      emea: "欧洲中东非洲",
      amer: "美洲",
    },
    reason: {
      no_overdue_invoice: "没有逾期发票",
      agent_reply_unavailable: "客服回复不可用",
      pending_customer_confirmation: "等待客户确认",
      reply_failed_metric_validation: "回复未通过指标验证",
      language_mismatch: "语言不匹配",
    },
  },
};

export function localizedValue(
  language: InteractionLanguage | undefined,
  value: unknown,
  group = "status"
): string {
  if (value == null || value === "") return "—";
  const raw = String(value);
  const normalized = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const baseCode = normalized.split(":")[0];
  const labels = VALUE_LABELS[language ?? "en-US"] ?? VALUE_LABELS["en-US"];
  return labels[group]?.[normalized] ?? labels[group]?.[baseCode] ?? labels.status?.[normalized] ?? raw.replace(/_/g, " ");
}

export function localizeRationale(
  language: InteractionLanguage | undefined,
  text: string | null | undefined
): string {
  if (!text) return "";
  if (language === "id-ID") {
    return text
      .replace(/overdue invoice exposure/gi, "eksposur invoice jatuh tempo")
      .replace(/declined payments/gi, "pembayaran ditolak")
      .replace(/declined payment/gi, "pembayaran ditolak")
      .replace(/account needs agent assist/gi, "akun perlu bantuan agen")
      .replace(/billing risk/gi, "risiko tagihan")
      .replace(/open issue/gi, "masalah terbuka")
      .replace(/overdue/gi, "jatuh tempo")
      .replace(/invoice/gi, "invoice")
      .replace(/payment/gi, "pembayaran");
  }
  if (language === "zh-CN") {
    return text
      .replace(/overdue invoice exposure/gi, "存在逾期发票风险")
      .replace(/declined payments/gi, "付款失败")
      .replace(/declined payment/gi, "付款失败")
      .replace(/account needs agent assist/gi, "账户需要客服协助")
      .replace(/billing risk/gi, "账单风险")
      .replace(/open issue/gi, "未关闭问题")
      .replace(/overdue/gi, "逾期")
      .replace(/invoice/gi, "发票")
      .replace(/payment/gi, "付款");
  }
  if (language !== "th-TH") return text;
  return text
    .replace(/overdue invoice exposure/gi, "มีใบแจ้งหนี้ค้างชำระ")
    .replace(/declined payments/gi, "มีการชำระเงินไม่สำเร็จ")
    .replace(/declined payment/gi, "มีการชำระเงินไม่สำเร็จ")
    .replace(/account needs agent assist/gi, "บัญชีนี้ต้องการความช่วยเหลือจากเจ้าหน้าที่")
    .replace(/billing risk/gi, "ความเสี่ยงด้านบิล")
    .replace(/open issue/gi, "ปัญหาที่ยังเปิดอยู่")
    .replace(/overdue/gi, "ค้างชำระ")
    .replace(/invoice/gi, "ใบแจ้งหนี้")
    .replace(/payment/gi, "การชำระเงิน");
}

export function localizeResolutionNote(
  language: InteractionLanguage | undefined,
  text: string | null | undefined
): string {
  if (!text) return "";
  if (language === "id-ID") {
    return text
      .replace(/issue_in_progress/gi, "masalah sedang diproses")
      .replace(/guided by Genie and account context/gi, "dipandu oleh Genie dan konteks akun")
      .replace(/overdue amount/gi, "jumlah jatuh tempo")
      .replace(/STATUS_CHANGED/gi, "status berubah")
      .replace(/IN_PROGRESS/gi, "sedang diproses")
      .replace(/overdue/gi, "jatuh tempo");
  }
  if (language === "zh-CN") {
    return text
      .replace(/issue_in_progress/gi, "问题处理中")
      .replace(/guided by Genie and account context/gi, "由 Genie 和账户上下文指导")
      .replace(/overdue amount/gi, "逾期金额")
      .replace(/STATUS_CHANGED/gi, "状态已变更")
      .replace(/IN_PROGRESS/gi, "处理中")
      .replace(/overdue/gi, "逾期");
  }
  if (language !== "th-TH") return text;
  return text
    .replace(/issue_in_progress/gi, "ปัญหากำลังดำเนินการ")
    .replace(/guided by Genie and account context/gi, "นำทางด้วย Genie และบริบทบัญชี")
    .replace(/overdue amount/gi, "ยอดค้างชำระ")
    .replace(/STATUS_CHANGED/gi, "เปลี่ยนสถานะ")
    .replace(/IN_PROGRESS/gi, "กำลังดำเนินการ")
    .replace(/overdue/gi, "ค้างชำระ");
}

export function localizedLanguageName(
  uiLanguage: InteractionLanguage | undefined,
  code: InteractionLanguage,
  fallback: string
): string {
  const labels: Partial<Record<InteractionLanguage, Record<InteractionLanguage, string>>> = {
    "th-TH": {
      "en-US": "อังกฤษ",
      "th-TH": "ไทย",
      "id-ID": "อินโดนีเซีย",
      "zh-CN": "จีนกลาง",
    },
    "id-ID": {
      "en-US": "Inggris",
      "th-TH": "Thai",
      "id-ID": "Indonesia",
      "zh-CN": "Mandarin",
    },
    "zh-CN": {
      "en-US": "英语",
      "th-TH": "泰语",
      "id-ID": "印尼语",
      "zh-CN": "普通话",
    },
  };
  return labels[uiLanguage ?? "en-US"]?.[code] ?? fallback;
}
