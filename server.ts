import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Support larger payload for medical report images / PDFs
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true, limit: "25mb" }));

// Lazy getter for GoogleGenAI instance
let aiClient: GoogleGenAI | null = null;
function getAIClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("GEMINI_API_KEY is not set in environment variables.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "",
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// 1. Health check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    time: new Date().toISOString(),
  });
});

// Helper for safe JSON parsing from Gemini text
function extractCleanJson(text: string): any {
  if (!text) return null;
  let clean = text.trim();
  // Remove markdown code fences if present
  clean = clean.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/\s*```$/, "");
  try {
    return JSON.parse(clean);
  } catch (err) {
    // Attempt to extract the first { ... } block
    const match = clean.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (innerErr) {
        console.error("JSON regex parse failed:", innerErr);
      }
    }
    console.error("Failed to parse JSON from AI response:", text);
    return null;
  }
}

// 2. Symptom Checker & Triage API
app.post("/api/triage", async (req, res) => {
  try {
    const { symptoms, duration, patientInfo, language = "en" } = req.body;

    if (!symptoms || typeof symptoms !== "string" || symptoms.trim().length === 0) {
      return res.status(400).json({
        error: language === "ur" ? "براہ کرم علامات درج کریں۔" : "Please describe your symptoms.",
      });
    }

    const ai = getAIClient();

    const systemPrompt = `You are ShifaTech AI, an advanced bilingual medical triage and health information assistant supporting English and Urdu.
STRICT MEDICAL SAFETY DIRECTIVES:
1. You are an AI-assisted triage tool ONLY. You MUST NEVER claim to provide a definitive medical diagnosis.
2. You must NEVER prescribe medication, recommend specific dosages, or tell a patient to start or stop prescribed medicines.
3. If red flag symptoms (e.g. crushing chest pain, severe shortness of breath, sudden one-sided weakness, severe uncontrolled bleeding, anaphylaxis signs, severe sudden confusion) are present, you MUST classify riskLevel as "EMERGENCY" and urge immediate local emergency medical service (EMS) contact.
4. If the provided information is too sparse or insufficient, you must explicitly state that you cannot reliably assess the situation.
5. Always use cautious medical language: "may indicate", "could be associated with", "possible concerns", "AI-assisted triage assessment".
6. Provide responses in the requested language (${language === "ur" ? "Urdu with clean Nastaliq/Arabic script text" : "English"}), but ensure the JSON keys match the exact required schema. Also include both English and Urdu text fields where indicated.

Return ONLY a valid JSON object with EXACTLY this structure:
{
  "riskLevel": "LOW" | "MODERATE" | "HIGH" | "EMERGENCY",
  "summary": "Concise summary in ${language === "ur" ? "Urdu" : "English"}",
  "summaryEn": "Concise summary in English",
  "summaryUr": "مختصر خلاصہ اردو میں",
  "possibleConcerns": ["Possible concern 1", "Possible concern 2"],
  "possibleConcernsUr": ["ممکنہ وجہ 1", "ممکنہ وجہ 2"],
  "recommendedNextStep": "Clear next step advice in ${language === "ur" ? "Urdu" : "English"}",
  "recommendedNextStepEn": "Clear next step advice in English",
  "recommendedNextStepUr": "اگلا مجوزہ قدم اردو میں",
  "warningSigns": ["Red flag warning sign 1", "Red flag warning sign 2"],
  "warningSignsUr": ["انتباہی علامت 1", "انتباہی علامت 2"],
  "doctorQuestions": ["Specific question for doctor 1", "Question 2"],
  "doctorQuestionsUr": ["ڈاکٹر سے پوچھنے کے سوالات 1", "سوال 2"],
  "clinicalNotes": "Objective clinical presentation summary for medical providers",
  "disclaimer": "ShifaTech AI provides AI-assisted health information and triage guidance only. It does not replace professional medical evaluation, diagnosis, or treatment."
}`;

    const userContent = `Patient Symptoms: ${symptoms}
Duration: ${duration || "Not specified"}
Patient Demographics & Context:
- Age bracket: ${patientInfo?.age || "Unspecified"}
- Biological sex: ${patientInfo?.gender || "Unspecified"}
- Existing health conditions: ${patientInfo?.conditions || "None reported"}
- Current medications: ${patientInfo?.medications || "None reported"}
- Pregnancy status: ${patientInfo?.isPregnant ? "Yes / Possible" : "No / Not applicable"}
Target Language: ${language === "ur" ? "Urdu" : "English"}`;

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: userContent,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        temperature: 0.2,
      },
    });

    const parsed = extractCleanJson(response.text || "");
    if (!parsed || !parsed.riskLevel) {
      // Safe fallback
      return res.json({
        riskLevel: "MODERATE",
        summary: language === "ur"
          ? "ہم آپ کی معلومات کا خودکار طریقے سے مکمل جائزہ نہیں لے سکے۔ براہ کرم مستند معالج سے رجوع فرمائیں۔"
          : "We could not reliably finalize an automated assessment from this input. Please consult a qualified healthcare professional.",
        summaryEn: "We could not reliably finalize an automated assessment from this input. Please consult a qualified healthcare professional.",
        summaryUr: "ہم آپ کی معلومات کا خودکار طریقے سے مکمل جائزہ نہیں لے سکے۔ براہ کرم مستند معالج سے رجوع فرمائیں۔",
        possibleConcerns: ["Unspecified symptom complex"],
        possibleConcernsUr: ["غیر واضح علامات کا مجموعہ"],
        recommendedNextStep: language === "ur" ? "طبی معالج یا کلینک سے رابطہ کریں" : "Contact a healthcare provider for physical evaluation",
        recommendedNextStepEn: "Contact a healthcare provider for physical evaluation",
        recommendedNextStepUr: "طبی معالج یا کلینک سے رابطہ کریں",
        warningSigns: [
          "Severe sudden pain or difficulty breathing",
          "Sudden neurological changes or confusion"
        ],
        warningSignsUr: [
          "اچانک شدید درد یا سانس لینے میں دشواری",
          "بے ہوشی یا شدید الجھن"
        ],
        doctorQuestions: ["What underlying causes should we investigate for these symptoms?"],
        doctorQuestionsUr: ["ان علامات کی کیا وجوہات ہو سکتی ہیں؟"],
        clinicalNotes: "Patient reported non-specific symptoms. Direct clinical evaluation advised.",
        disclaimer: "ShifaTech AI provides AI-assisted health information only. It does not replace a doctor.",
      });
    }

    res.json(parsed);
  } catch (error: any) {
    console.error("Error in /api/triage:", error);
    res.status(500).json({
      error: "Unable to complete symptom triage at this moment.",
      details: error?.message || "Internal server error",
    });
  }
});

// 3. Medical Report Scanner API
app.post("/api/analyze-report", async (req, res) => {
  try {
    const { fileData, mimeType, fileName, language = "en" } = req.body;

    if (!fileData) {
      return res.status(400).json({ error: "Missing document/image data" });
    }

    const ai = getAIClient();

    const systemPrompt = `You are ShifaTech AI Medical Report Scanner.
You analyze lab test reports, radiology scans, blood work (CBC, Lipid panel, LFT, RFT, HbA1c, etc.), pathology, or diagnostic documents.

IMPORTANT MEDICAL SAFETY & EXTRACTION RULES:
1. Extract REAL findings ONLY. Never invent or hallucinate test names, values, or reference ranges.
2. If any section or value is blurred, clipped, or unreadable, set its status or note as "Unable to reliably read this section."
3. When a reference range is present, compare the value and indicate whether it is within range, elevated, or low.
4. DO NOT diagnose a disease from a single out-of-range value. Always frame findings neutrally: "Value appears elevated relative to the laboratory reference range; correlation with clinical symptoms and doctor review is required."
5. Extract metadata: test names, values, units, reference ranges, test date, and important observations.
6. Provide output in structured JSON. Include bilingual representations (English and Urdu) for patient comprehension.

Output schema:
{
  "reportTitle": "Title or type of test (e.g. Complete Blood Count / Lipid Profile)",
  "reportDate": "Report date or 'Not indicated'",
  "laboratoryName": "Laboratory name if visible or 'Not visible'",
  "overallSummary": "Clear plain-language overview for the patient in ${language === "ur" ? "Urdu" : "English"}",
  "overallSummaryEn": "Clear plain-language overview in English",
  "overallSummaryUr": "مریض کے لیے رپورٹ کا سادہ اور واضح خلاصہ اردو میں",
  "extractedParameters": [
    {
      "testName": "Hemoglobin / Hb",
      "value": "13.2",
      "unit": "g/dL",
      "referenceRange": "13.5 - 17.5 g/dL",
      "status": "NORMAL" | "HIGH" | "LOW" | "UNCLEAR",
      "interpretation": "Slightly below standard male reference range or within normal female range.",
      "interpretationUr": "معیاری حد کے قریب یا معمولی کم۔"
    }
  ],
  "potentialFindings": [
    "Key finding 1 with neutral clinical context",
    "Key finding 2"
  ],
  "potentialFindingsUr": [
    "اہم مشاہدہ 1 بغیر کسی خودکار تشخیص کے",
    "اہم مشاہدہ 2"
  ],
  "doctorQuestions": [
    "Questions patient can ask their doctor about these findings"
  ],
  "doctorQuestionsUr": [
    "معالج سے رپورٹ کے بارے میں پوچھنے کے لیے مفید سوالات"
  ],
  "unreadableSections": ["Notes on any blurred or illegible sections"],
  "disclaimer": "Diagnostic reports require clinical correlation by the ordering physician. Single values out of range do not constitute a diagnosis."
}`;

    const cleanBase64 = fileData.replace(/^data:[^;]+;base64,/, "");

    const documentPart = {
      inlineData: {
        mimeType: mimeType || "image/jpeg",
        data: cleanBase64,
      },
    };

    const textPart = {
      text: `Please scan and extract all diagnostic test data, reference ranges, and observations from this medical document (${fileName || "Uploaded report"}). Target output language: ${language}.`,
    };

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: { parts: [documentPart, textPart] },
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    });

    const parsed = extractCleanJson(response.text || "");
    if (!parsed) {
      return res.status(502).json({
        error: "Unable to parse the medical report format reliably. Please ensure the document is clear and well-lit.",
      });
    }

    res.json(parsed);
  } catch (error: any) {
    console.error("Error in /api/analyze-report:", error);
    res.status(500).json({
      error: "Error processing report document.",
      details: error?.message || "Internal error",
    });
  }
});

// 4. Handwritten Prescription Reader API
app.post("/api/analyze-prescription", async (req, res) => {
  try {
    const { fileData, mimeType, fileName, language = "en" } = req.body;

    if (!fileData) {
      return res.status(400).json({ error: "Missing prescription image" });
    }

    const ai = getAIClient();

    const systemPrompt = `You are ShifaTech AI Handwritten Prescription Reader.
You read handwritten and printed medical prescriptions using advanced OCR and vision reasoning.

CRITICAL MEDICAL SAFETY RULES:
1. NEVER guess or fabricate a medication name if the handwriting is unclear, smudged, or ambiguous.
2. For any uncertain handwriting or illegible dosage/frequency, you MUST clearly flag it: "Text unclear — please verify this medicine with a pharmacist or doctor."
3. NEVER recommend changing dosages or stopping prescribed medications.
4. Include explicit instructions: "Always have this prescription verified by a licensed pharmacist prior to taking any medication."
5. Extract readable medicine names, strength/dose, frequency (e.g. once daily, twice daily), duration, and special instructions.
6. Provide structured output in JSON.

Output JSON schema:
{
  "doctorHospitalName": "Doctor or clinic name if visible, or 'Not visible'",
  "prescriptionDate": "Date or 'Not indicated'",
  "overallNotes": "Patient-friendly overview in ${language === "ur" ? "Urdu" : "English"}",
  "overallNotesEn": "Patient-friendly overview in English",
  "overallNotesUr": "مریض کے لیے نسخے کی مجموعی معلومات اردو میں",
  "medications": [
    {
      "medicineName": "Amoxicillin / Clavulanate (or 'Unclear')",
      "strength": "625 mg",
      "frequency": "Twice daily after food (BID PC)",
      "duration": "5 days",
      "specialInstructions": "Take with meals",
      "specialInstructionsUr": "کھانے کے بعد لیں",
      "legibility": "CLEAR" | "MODERATE" | "UNCLEAR",
      "pharmacistVerificationNotice": "Text unclear — please verify this medicine with a pharmacist or doctor." (if legibility is UNCLEAR or MODERATE, else empty string or normal notice)
    }
  ],
  "lifestyleOrGeneralAdvice": [
    "General non-pharmacological advice found on prescription like 'drink plenty of fluids', 'rest'"
  ],
  "lifestyleOrGeneralAdviceUr": [
    "نسخے میں لکھی عمومی ہدایات (مثلاً آرام کریں، پانی زیادہ پئیں)"
  ],
  "unclearItemsCount": 0,
  "importantSafetyWarning": "Never adjust dosages or stop prescribed medication on your own. Always present the original prescription to a licensed pharmacist.",
  "importantSafetyWarningUr": "اپنی مرضی سے ادویات کی خوراک تبدیل نہ کریں اور نہ ہی بند کریں۔ ادویات لینے سے پہلے ہمیشہ مستند فارماسسٹ کو اصلی نسخہ دکھائیں۔"
}`;

    const cleanBase64 = fileData.replace(/^data:[^;]+;base64,/, "");

    const imagePart = {
      inlineData: {
        mimeType: mimeType || "image/jpeg",
        data: cleanBase64,
      },
    };

    const textPart = {
      text: `Carefully read this prescription photo (${fileName || "Prescription"}). Identify legible medications and explicitly mark any uncertain handwriting. Target language: ${language}.`,
    };

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: { parts: [imagePart, textPart] },
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    });

    const parsed = extractCleanJson(response.text || "");
    if (!parsed) {
      return res.status(502).json({
        error: "Unable to parse prescription. Please ensure the photo has good lighting and is focused.",
      });
    }

    res.json(parsed);
  } catch (error: any) {
    console.error("Error in /api/analyze-prescription:", error);
    res.status(500).json({
      error: "Error processing prescription image.",
      details: error?.message || "Internal error",
    });
  }
});

// 5. Audio Transcription API (Pre-recorded or client-recorded audio fallback)
app.post("/api/transcribe-audio", async (req, res) => {
  try {
    const { audioData, mimeType = "audio/webm", language = "en" } = req.body;
    if (!audioData) {
      return res.status(400).json({ error: "Missing audio data" });
    }

    const ai = getAIClient();
    const cleanBase64 = audioData.replace(/^data:[^;]+;base64,/, "");

    const audioPart = {
      inlineData: {
        mimeType: mimeType,
        data: cleanBase64,
      },
    };

    const promptText = language === "ur"
      ? "براہ کرم اس آڈیو میں مریض کی بیان کردہ علامات کو ہو بہو اردو متن میں تحریر کریں۔ صرف بولے گئے الفاظ لکھیں، کوئی اضافی تبصرہ نہ کریں۔"
      : "Transcribe the patient's spoken symptoms accurately in text. Return only the transcription without extra commentary.";

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: { parts: [audioPart, { text: promptText }] },
    });

    const transcription = response.text?.trim() || "";
    res.json({ transcription });
  } catch (error: any) {
    console.error("Error in /api/transcribe-audio:", error);
    res.status(500).json({
      error: "Unable to transcribe audio recording.",
      details: error?.message || "Audio processing failed",
    });
  }
});

// Vite middleware / production static files
async function setupApp() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`ShifaTech AI server running on port ${PORT}`);
  });
}

setupApp();
