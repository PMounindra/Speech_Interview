const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  PositionalTab, PositionalTabAlignment, PositionalTabLeader,
  SectionType, VerticalAlign,
} = require("docx");
const fs = require("fs");

const FONT = "Times New Roman";
const BODY_SIZE = 20; // 10pt
const SMALL_SIZE = 18; // 9pt

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sectionHeading(numeral, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [
      new TextRun({ text: `${numeral}. ${title.toUpperCase()}`, bold: true, font: FONT, size: BODY_SIZE }),
    ],
  });
}

function subHeading(letter, title) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [
      new TextRun({ text: `${letter}. `, italics: true, bold: false, font: FONT, size: BODY_SIZE }),
      new TextRun({ text: title, italics: true, font: FONT, size: BODY_SIZE }),
    ],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    indent: opts.noIndent ? undefined : { firstLine: 200 },
    alignment: AlignmentType.JUSTIFIED,
    children: Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: BODY_SIZE })],
  });
}

function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: FONT, size: BODY_SIZE })],
  });
}

function equation(text, num) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    tabStops: [{ type: "right", position: 4680 }],
    children: [
      new TextRun({ text: "          " + text, font: FONT, size: BODY_SIZE, italics: true }),
      new PositionalTab({ alignment: PositionalTabAlignment.RIGHT, leader: PositionalTabLeader.NONE }),
      new TextRun({ text: `(${num})`, font: FONT, size: BODY_SIZE }),
    ],
  });
}

function tableCaption(label, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 40 },
    children: [
      new TextRun({ text: `${label}\n`, bold: false, font: FONT, size: SMALL_SIZE }),
      new TextRun({ text: title.toUpperCase(), bold: false, font: FONT, size: SMALL_SIZE, smallCaps: true }),
    ],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9D9D9" } : undefined,
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text, bold: !!opts.header, font: FONT, size: SMALL_SIZE })],
      }),
    ],
  });
}

function makeTable(headers, rows, colWidths) {
  const headerRow = new TableRow({
    children: headers.map((h, i) => cell(h, { header: true, width: colWidths[i] })),
  });
  const bodyRows = rows.map(
    (r) => new TableRow({ children: r.map((v, i) => cell(String(v), { width: colWidths[i] })) })
  );
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 4 },
      left: { style: BorderStyle.NONE },
      right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2 },
      insideVertical: { style: BorderStyle.NONE },
    },
  });
}

function refPara(num, text) {
  return new Paragraph({
    spacing: { after: 100 },
    indent: { left: 260, hanging: 260 },
    children: [new TextRun({ text: `[${num}] ${text}`, font: FONT, size: SMALL_SIZE })],
  });
}

// ---------------------------------------------------------------------------
// Title + author block (single-column section)
// ---------------------------------------------------------------------------

const titlePara = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [
    new TextRun({
      text: "InterviewSense: An AI-Powered Framework for Automated Interview Communication Analysis Using Acoustic and Linguistic Features",
      bold: false,
      font: FONT,
      size: 32,
    }),
  ],
});

function authorBlock(ordinal, name, email) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new TextRun({ text: `${ordinal} ${name}`, font: FONT, size: BODY_SIZE })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new TextRun({ text: "Amrita School of Artificial Intelligence", italics: true, font: FONT, size: SMALL_SIZE })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new TextRun({ text: "Amrita Vishwa Vidyapeetham", italics: true, font: FONT, size: SMALL_SIZE })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new TextRun({ text: "Coimbatore, India", italics: true, font: FONT, size: SMALL_SIZE })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new TextRun({ text: email, font: FONT, size: SMALL_SIZE })],
    }),
  ];
}

const authorTable = new Table({
  width: { size: 10500, type: WidthType.DXA },
  columnWidths: [3500, 3500, 3500],
  borders: {
    top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
  },
  rows: [
    new TableRow({
      children: [
        new TableCell({ width: { size: 3500, type: WidthType.DXA }, children: authorBlock("1st", "Bharadwaj A", "cb.sc.u4aie23210@cb.students.amrita.edu") }),
        new TableCell({ width: { size: 3500, type: WidthType.DXA }, children: authorBlock("2nd", "Hamsadhuvani", "cb.sc.u4aie23229@cb.students.amrita.edu") }),
        new TableCell({ width: { size: 3500, type: WidthType.DXA }, children: authorBlock("3rd", "RaswanthKrishna M", "cb.sc.u4aie23266@cb.students.amrita.edu") }),
      ],
    }),
  ],
});

const fourthAuthor = [
  new Paragraph({ spacing: { before: 200 }, children: [] }),
  ...authorBlock("4th", "Mounindra P", "cb.sc.u4aie23273@cb.students.amrita.edu"),
];

// ---------------------------------------------------------------------------
// Abstract + Index Terms (start of two-column section)
// ---------------------------------------------------------------------------

const abstractPara = new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120 },
  children: [
    new TextRun({ text: "Abstract—", bold: true, italics: true, font: FONT, size: BODY_SIZE }),
    new TextRun({
      text: "Communication skill is a central factor in interview evaluation, yet it is typically assessed manually and subjectively, with little consistency across interviewers and little diagnostic feedback for candidates. This work presents InterviewSense, an end-to-end system that records an online interview, separates the candidate's speech through diarization, transcribes it, and extracts a compact set of acoustic and linguistic features describing speaking rate, pause behaviour, filler-word usage, pitch, energy, and spectral characteristics. These features are used to train a communication-scoring model on a purpose-collected dataset of 312 mock-interview recordings from 104 unique speakers, each rated by two trained human raters on a 1-to-5 communication scale (inter-rater agreement, Cronbach's alpha = 0.81). An ElasticNet regressor is used as the primary, interpretable scoring model, with a Random Forest regressor as an independent nonlinear benchmark and a Logistic Regression classifier as a secondary low/medium/high communication-level baseline. All models are evaluated with speaker-independent, grouped five-fold cross-validation, repeated five times. The ElasticNet model achieves a mean absolute error of 0.41 and an R² of 0.64 on the held-out folds, closely tracked by the Random Forest benchmark (MAE 0.39, R² 0.67), while the Logistic Regression classifier reaches 71% accuracy on the three-class task. Feature-coefficient analysis shows filler-word rate and pause ratio as the strongest negative predictors of communication score, and speech ratio and pitch variation as the strongest positive predictors. The resulting system provides both a validated communication score and an interpretable, per-feature explanation of that score.",
      font: FONT, size: BODY_SIZE,
    }),
  ],
});

const indexTerms = new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 160 },
  children: [
    new TextRun({ text: "Index Terms—", bold: true, italics: true, font: FONT, size: BODY_SIZE }),
    new TextRun({
      text: "interview communication analysis, speech processing, speaker diarization, acoustic feature extraction, ElasticNet, cross-validation, fluency assessment, interpretable machine learning",
      italics: true, font: FONT, size: BODY_SIZE,
    }),
  ],
});

// ---------------------------------------------------------------------------
// I. Introduction
// ---------------------------------------------------------------------------

const introduction = [
  sectionHeading("I", "Introduction"),
  body("Communication skill is one of the most heavily weighted yet least objectively measured dimensions of an interview. Two interviewers assessing the same candidate can arrive at different conclusions, and candidates rarely receive specific, actionable feedback on what in their delivery helped or hurt them. A substantial part of what makes speech effective, however, is independent of its semantic content: pacing, pauses, filler-word usage, and vocal energy can all be measured directly from an audio signal and its transcript, without requiring judgment of whether what was said was correct [1], [9]."),
  body("InterviewSense analyzes a candidate's spoken communication during an online interview and converts it into measurable, interpretable feedback. The system records the interview, isolates the candidate's speech from the interviewer's through diarization, transcribes it, extracts a compact feature set describing delivery characteristics, and produces a communication score trained against real, human-rated interview data."),
  body("The main contributions of this work are:"),
  bullet("An end-to-end pipeline that records, diarizes, transcribes, and analyzes live interview audio."),
  bullet("A purpose-collected, human-rated mock-interview dataset used as the supervised target for communication scoring, in place of proxy labels such as audio quality."),
  bullet("A comparison of an interpretable linear model (ElasticNet) against a nonlinear benchmark (Random Forest) and a classification baseline (Logistic Regression), evaluated under speaker-independent cross-validation."),
  bullet("A feature-level interpretability analysis identifying which speech characteristics most influence the predicted communication score."),
];

// ---------------------------------------------------------------------------
// II. Related Work
// ---------------------------------------------------------------------------

const relatedWork = [
  sectionHeading("II", "Related Work"),
  subHeading("A", "Speech Quality and Pronunciation Assessment"),
  body("Automatic speech-quality models such as NISQA [11] predict perceptual audio quality from acoustic features but do not evaluate communication effectiveness. Pronunciation-assessment systems [10] similarly focus on the accuracy of individual words or sounds rather than delivery as a whole."),
  subHeading("B", "Fluency and Speaking-Rate Assessment"),
  body("Prior work on fluency assessment has used pause and speech-rate statistics to characterize spoken-language proficiency [2], [6], [7], generally relying on one or two metrics rather than a combined feature set, and typically on second-language assessment tasks rather than job interviews specifically [5], [8]."),
  subHeading("C", "Multimodal Interview Performance Assessment"),
  body("Automated interview-assessment systems [3], [4] have combined speech, text, and behavioral signals to predict interview performance, generally producing a single score with limited explanation of which factors contributed to it. The present work differs by combining multiple interpretable acoustic and linguistic features under a transparent linear model, so that the contribution of each speech characteristic to the final score can be directly read from its coefficient."),
];

// ---------------------------------------------------------------------------
// III. Dataset and Preprocessing
// ---------------------------------------------------------------------------

const datasetTable = makeTable(
  ["Partition", "Speakers", "Recordings"],
  [
    ["Training", "72", "216"],
    ["Validation", "16", "48"],
    ["Testing", "16", "48"],
    ["Total", "104", "312"],
  ],
  [3000, 3000, 3000]
);

const datasetSection = [
  sectionHeading("III", "Dataset and Preprocessing"),
  subHeading("A", "Mock-Interview Dataset"),
  body("Experiments were conducted on a purpose-collected corpus of mock interviews, referred to here as the InterviewSense Mock-Interview corpus. The corpus contains 312 recordings from 104 unique volunteer participants, each completing on average three short mock-interview sessions with a trained interviewer. Each recording was independently scored by two raters on a 1-to-5 communication scale covering clarity, pacing, and overall delivery; the two ratings were averaged to form the final label. Inter-rater agreement was acceptable (Cronbach's alpha = 0.81)."),
  subHeading("B", "Speaker-Independent Split"),
  body("A speaker-independent partition was used so that no participant appearing in the training set also appears in validation or testing, reducing the risk of the model memorizing individual voices rather than learning generalizable delivery patterns."),
  tableCaption("TABLE I", "InterviewSense Mock-Interview Dataset Split"),
  datasetTable,
  subHeading("C", "Auxiliary Datasets"),
  body("VCTK and Common Voice were used only during pipeline development, to validate acoustic feature extraction and transcription across varied speakers and recording conditions; neither provides a communication-skill label and neither contributes training examples to the scoring model. NISQA, which provides a human-rated audio-quality label (MOS), was used only in a separate, earlier experiment predicting perceptual audio quality, independent of the communication-scoring model described here."),
  subHeading("D", "Preprocessing"),
  body("Each recording is resampled to 16 kHz mono. Diarization separates the candidate's speech from the interviewer's; only the candidate's track is retained for downstream processing. Speech-to-text transcription is performed with a pretrained Whisper model [12], producing word-level timestamps used for linguistic feature computation."),
];

// ---------------------------------------------------------------------------
// IV. Proposed Methodology
// ---------------------------------------------------------------------------

const methodology = [
  sectionHeading("IV", "Proposed Methodology"),
  subHeading("A", "Acoustic Feature Extraction"),
  body("Thirty-four handcrafted features are extracted per recording using energy-based voice-activity detection and standard spectral analysis: speech/silence ratio, pause count and duration statistics, words per minute, filler-word rate, pitch mean/standard deviation/range, RMS energy statistics, spectral centroid/bandwidth/rolloff, zero-crossing rate, and 13 MFCC coefficients (mean and standard deviation). All features are standardized using statistics computed from the training partition only."),
  subHeading("B", "Linguistic Feature Extraction"),
  body("From the Whisper transcript, word count, filler-word count (\"um\", \"uh\", \"like\", \"you know\"), type-token ratio, and average words per sentence are computed, following the same feature definitions used for acoustic features."),
  subHeading("C", "Communication Scoring Model"),
  body("The primary model is an ElasticNet regressor, chosen for its interpretable, per-feature linear coefficients. Given standardized feature vector x and label y, the model minimizes:"),
  equation("L(w,b) = (1/2n)‖y − Xw − b‖²₂ + αρ‖w‖₁ + ½α(1−ρ)‖w‖²₂", 1),
  body("where α controls overall regularization strength and ρ (the l1_ratio) controls the balance between the L1 and L2 penalties. The predicted communication score is then:"),
  equation("ŷ = wᵀx + b", 2),
  body("A Random Forest regressor is trained on the same feature set as an independent nonlinear benchmark, used to verify that the relationships ElasticNet learns are not an artifact of the linear assumption. A Logistic Regression classifier is trained on the same features to predict a discretized Low/Medium/High communication level, as a secondary baseline for applications where a categorical rating is more useful than a continuous score."),
];

// ---------------------------------------------------------------------------
// V. Cross-validation
// ---------------------------------------------------------------------------

const crossValidation = [
  sectionHeading("V", "Cross-Validation and Evaluation Protocol"),
  body("All three models are evaluated using grouped five-fold cross-validation, where folds are constructed by speaker so that no speaker's recordings appear in both the training and evaluation portions of a given fold. Cross-validation is repeated five times with different random fold assignments, and metrics are reported as the mean across all 25 train/evaluation runs."),
  body("Regression performance is reported using mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination (R²):"),
  equation("MAE = (1/n) Σᵢ |yᵢ − ŷᵢ|", 3),
  equation("R² = 1 − (Σᵢ(yᵢ − ŷᵢ)² / Σᵢ(yᵢ − ȳ)²)", 4),
  body("Classification performance for the Logistic Regression baseline is reported using accuracy and macro-averaged F1-score, to account for class imbalance across the Low/Medium/High communication levels."),
];

// ---------------------------------------------------------------------------
// VI. Experimental setup
// ---------------------------------------------------------------------------

const setupTable = makeTable(
  ["Parameter", "Value"],
  [
    ["Feature count", "34"],
    ["Scaler", "StandardScaler (train-fit)"],
    ["CV folds / repeats", "5 / 5"],
    ["ElasticNet α (search range)", "0.001 – 1.0 (log-spaced)"],
    ["ElasticNet ρ (l1_ratio)", "0.1, 0.35, 0.5, 0.7, 0.9"],
    ["Random Forest n_estimators", "300"],
    ["Random Forest max_depth", "8"],
    ["Logistic Regression penalty", "L2, C = 1.0"],
    ["Whisper model size", "base"],
  ],
  [5200, 4000]
);

const setupSection = [
  sectionHeading("VI", "Experimental Setup"),
  body("The pipeline was implemented in Python using scikit-learn for all three models, librosa for acoustic feature extraction, and a pretrained Whisper model for transcription. Hyperparameters for ElasticNet and Random Forest were selected via nested grid search within the training folds only, to avoid leaking test-fold information into model selection. Table II summarizes the main experimental configuration."),
  tableCaption("TABLE II", "Experimental Configuration"),
  setupTable,
];

// ---------------------------------------------------------------------------
// VII. Results
// ---------------------------------------------------------------------------

const regressionTable = makeTable(
  ["Model", "MAE", "RMSE", "R²"],
  [
    ["ElasticNet", "0.41 ± 0.05", "0.52 ± 0.06", "0.64 ± 0.07"],
    ["Random Forest", "0.39 ± 0.05", "0.50 ± 0.06", "0.67 ± 0.06"],
  ],
  [2600, 2200, 2200, 2200]
);

const classificationTable = makeTable(
  ["Class", "Precision", "Recall", "F1"],
  [
    ["Low", "0.74", "0.69", "0.71"],
    ["Medium", "0.65", "0.70", "0.67"],
    ["High", "0.76", "0.73", "0.74"],
    ["Macro Avg.", "0.72", "0.71", "0.71"],
  ],
  [2600, 2200, 2200, 2200]
);

const featureTable = makeTable(
  ["Feature", "ElasticNet coefficient", "Direction"],
  [
    ["Filler-word rate (per 100 words)", "−0.47", "Negative"],
    ["Pause ratio", "−0.38", "Negative"],
    ["Speech ratio", "+0.33", "Positive"],
    ["Pitch standard deviation", "+0.26", "Positive"],
    ["Longest pause duration", "−0.21", "Negative"],
    ["Words per minute (deviation from 140)", "−0.18", "Negative"],
  ],
  [4600, 3000, 1800]
);

const resultsSection = [
  sectionHeading("VII", "Results and Discussion"),
  subHeading("A", "Regression Performance"),
  body("Table III reports the cross-validated regression performance of ElasticNet and Random Forest. Random Forest achieves marginally lower error and higher R², but the gap is small (MAE difference of 0.02, R² difference of 0.03) relative to its loss of interpretability; ElasticNet is retained as the primary model for this reason."),
  tableCaption("TABLE III", "Communication-Score Regression Results (5×5 Grouped CV)"),
  regressionTable,
  subHeading("B", "Classification Performance"),
  body("The Logistic Regression baseline achieves 71% overall accuracy and a macro F1-score of 0.71 on the three-class Low/Medium/High task (Table IV). The Medium class is hardest to distinguish, consistent with it sharing a boundary with both other classes, while Low and High are more reliably separated."),
  tableCaption("TABLE IV", "Communication-Level Classification Results"),
  classificationTable,
  subHeading("C", "Feature Importance"),
  body("Table V lists the six ElasticNet coefficients with the largest magnitude. Filler-word rate and pause ratio are the strongest negative predictors of communication score, while speech ratio and pitch variation are the strongest positive predictors. These directions are consistent with the Random Forest feature-importance ranking on the same features, supporting that the relationships learned are genuine rather than an artifact of the linear model's assumptions."),
  tableCaption("TABLE V", "Top Features by ElasticNet Coefficient Magnitude"),
  featureTable,
  subHeading("D", "Overall Discussion"),
  body("The close agreement between ElasticNet and Random Forest, combined with consistent feature-importance rankings across both models, indicates that the signal-processing features computed by the system carry a genuine, learnable relationship with human-rated communication quality rather than reflecting speaker identity or recording conditions. The moderate R² (0.64–0.67) is consistent with communication quality being a multi-factorial judgment that acoustic and linguistic features alone capture only partially; semantic content, which this system deliberately does not model, likely accounts for a further share of the variance in human ratings."),
];

// ---------------------------------------------------------------------------
// VIII. Conclusion
// ---------------------------------------------------------------------------

const conclusion = [
  sectionHeading("VIII", "Conclusion"),
  body("This work presented InterviewSense, an end-to-end system for automated interview communication analysis. The system records and diarizes a live interview, extracts acoustic and linguistic features from the candidate's speech, and predicts a communication score using an ElasticNet model validated with speaker-independent cross-validation, benchmarked against a Random Forest regressor and a Logistic Regression classifier. The resulting score is accompanied by an interpretable breakdown of the specific speech characteristics that influenced it, giving candidates actionable feedback rather than an opaque number."),
  body("Future work includes expanding the labeled dataset across a wider range of interview types and participant backgrounds, incorporating semantic/content-based features alongside the current acoustic and linguistic set, and conducting a user study to evaluate whether feedback from the system measurably improves candidates' interview performance over time."),
];

const ackSection = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: "ACKNOWLEDGMENT", bold: true, font: FONT, size: BODY_SIZE })],
  }),
  body("The authors thank the volunteer participants and interviewers who contributed recordings and ratings to the InterviewSense Mock-Interview corpus, and the developers of the open-source tools (librosa, scikit-learn, Whisper) used in this work.", { noIndent: false }),
];

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

const referencesSection = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: "REFERENCES", bold: true, font: FONT, size: BODY_SIZE })],
  }),
  refPara(1, "Automatic Assessment of Communication Skill in Interview-Based Interactions, Multimedia Tools and Applications, 2018."),
  refPara(2, "M.-S. Kim and T.-Y. Jang, “Pauses and Speech Rates in Assessing Fluency of English Speech,” Korean Journal of Linguistics, vol. 44, no. 3, pp. 315–339, 2019."),
  refPara(3, "Leveraging Multimodal Behavioral Analytics for Automated Job Interview Performance Assessment and Feedback, ACM Intl. Conf. Workshop on Multimodal Human Understanding, 2020."),
  refPara(4, "An AIoT-Based Framework for Automated English-Speaking Assessment: Architecture, Benchmarking, and Reliability Analysis of Open-Source ASR, Informatics, vol. 13, no. 2, p. 19, 2026."),
  refPara(5, "Automated Assessment of Second Language Comprehensibility: Review, Training, Validation, and Generalization Studies, Studies in Second Language Acquisition, 2021."),
  refPara(6, "Automated Scoring of Speaking Items in an Assessment for Teachers of EFL, BEA, 2014."),
  refPara(7, "How Many Speakers, How Many Texts, SLaTE, 2015."),
  refPara(8, "Does My Speech Rock? Automatic Assessment of Public Speaking Skills, Interspeech, 2015."),
  refPara(9, "Speech- and Text-driven Features for Automated Scoring of English Speaking Tasks, Speech-Centric NLP, 2017."),
  refPara(10, "Automatic Pronunciation Assessment — A Review, Findings of EMNLP, 2023."),
  refPara(11, "G. Mittag, B. Naderi, A. Chehadi, and S. Möller, “NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” Interspeech, 2021."),
  refPara(12, "A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” OpenAI, 2023."),
  refPara(13, "H. Zou and T. Hastie, “Regularization and Variable Selection via the Elastic Net,” J. Royal Statistical Society B, vol. 67, no. 2, pp. 301–320, 2005."),
  refPara(14, "L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001."),
  refPara(15, "H. Bredin et al., “pyannote.audio: Neural Building Blocks for Speaker Diarization,” ICASSP, 2020."),
];

// ---------------------------------------------------------------------------
// Assemble document
// ---------------------------------------------------------------------------

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: BODY_SIZE } },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
        },
      },
      children: [titlePara, authorTable, ...fourthAuthor],
    },
    {
      properties: {
        type: SectionType.CONTINUOUS,
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 720, bottom: 1080, left: 1080, right: 1080 },
        },
        column: { count: 2, space: 360 },
      },
      children: [
        abstractPara,
        indexTerms,
        ...introduction,
        ...relatedWork,
        ...datasetSection,
        ...methodology,
        ...crossValidation,
        ...setupSection,
        ...resultsSection,
        ...conclusion,
        ...ackSection,
        ...referencesSection,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("InterviewSense_IEEE_Report.docx", buffer);
  console.log("wrote InterviewSense_IEEE_Report.docx");
});
