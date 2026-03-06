# Integration Guide — Hotel Gym Image Analyzer

This guide is for the .NET developer integrating the Gemini Vision gym analyzer
into the existing backend.

---

## 1. Getting a Gemini API Key

The analyzer uses **Google Gemini Vision API**. Follow these steps to get an API key:

### Option A — Google AI Studio (Fastest, recommended for testing)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API key** → **Create API key**
4. Copy the key — it starts with `AIza...`

> Free tier: 15 requests/minute, 1 million tokens/day.
> This is enough for testing; upgrade to paid for production.

### Option B — Google Cloud (Recommended for production)

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable **Generative Language API**:
   - APIs & Services → Enable APIs → search "Generative Language API" → Enable
3. Create a service account:
   - IAM & Admin → Service Accounts → Create
   - Grant role: **Vertex AI User** or **AI Platform Developer**
4. Create a key: Service Account → Keys → Add Key → JSON → Download
5. Or use API key directly:
   - APIs & Services → Credentials → Create Credentials → API Key
6. (Optional) Restrict the key to "Generative Language API" for security

**What we need from you:** Just the API key string. Store it in your environment as
`GEMINI_API_KEY`. Never commit it to source control.

---

## 2. API Endpoint

### POST `/analyze`

Send one or more gym image URLs and receive structured JSON with detected attributes.

**Request**

```json
POST /analyze
Content-Type: application/json

{
  "gymId": "hotel-456",
  "imageUrls": [
    "https://cdn-media.hotelgyms.com/images/gym/optimized/5ce4cb5b.jpg",
    "https://cdn-media.hotelgyms.com/images/gym/optimized/e79f6b45.jpg"
  ]
}
```

| Field       | Type          | Required | Notes                              |
|-------------|---------------|----------|------------------------------------|
| `gymId`     | string        | No       | Your reference ID, echoed back     |
| `imageUrls` | string array  | Yes      | 1–20 publicly accessible URLs      |

**Response**

```json
HTTP 200 OK
Content-Type: application/json

{
  "gymId": "hotel-456",
  "processingTimeMs": 2341,
  "analysis": {
    "experience": {
      "equipmentBrand": { "value": "Technogym", "confidence": 0.92 },
      "waterOption":    { "value": "Water Station", "confidence": 0.87 },
      "hasTowels":      { "value": true, "confidence": 0.78 }
    },
    "cardio": {
      "hasPeloton": { "value": false, "confidence": 0.95 }
    },
    "imageCount": 2,
    "analysisNotes": null
  }
}
```

**Field reference**

| Path | Type | Values |
|------|------|--------|
| `experience.equipmentBrand.value` | string or null | Brand name (e.g. "Technogym", "Matrix", "Precor") or null if undetectable |
| `experience.waterOption.value` | string | `"Bottled Water"` / `"Water Station"` / `"None"` / `"Unknown"` |
| `experience.hasTowels.value` | bool or null | `true` / `false` / `null` (indeterminate) |
| `cardio.hasPeloton.value` | bool or null | `true` / `false` / `null` (indeterminate) |
| `*.confidence` | float 0.0–1.0 | 1.0 = certain, 0.5 = uncertain, null value means < 0.4 confidence |
| `imageCount` | int | Number of images actually analyzed |
| `analysisNotes` | string or null | Set when image quality is poor |

**Error responses**

| Status | Meaning |
|--------|---------|
| 422 | Validation error — bad URLs, 0 images, or >20 images |
| 500 | Gemini API error — check server logs |

---

## 3. C# Integration Example

### 3.1 — Call the Python service from .NET

If you're running the Python FastAPI service as a sidecar or microservice:

```csharp
using System.Net.Http.Json;
using System.Text.Json;

public record FieldWithConfidence<T>(T Value, double Confidence);

public record GymExperience(
    FieldWithConfidence<string?> EquipmentBrand,
    FieldWithConfidence<string>  WaterOption,
    FieldWithConfidence<bool?>   HasTowels
);

public record GymCardio(
    FieldWithConfidence<bool?> HasPeloton
);

public record GymAnalysis(
    GymExperience Experience,
    GymCardio     Cardio,
    int           ImageCount,
    string?       AnalysisNotes
);

public record AnalyzeResponse(
    string?     GymId,
    GymAnalysis Analysis,
    int         ProcessingTimeMs
);

public class GymAnalyzerClient
{
    private readonly HttpClient _http;

    public GymAnalyzerClient(HttpClient http)
    {
        _http = http;
        _http.BaseAddress = new Uri("http://localhost:8000");
    }

    public async Task<AnalyzeResponse?> AnalyzeAsync(
        string gymId,
        IEnumerable<string> imageUrls,
        CancellationToken ct = default)
    {
        var payload = new { gymId, imageUrls };
        var response = await _http.PostAsJsonAsync("/analyze", payload, ct);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync<AnalyzeResponse>(
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true },
            ct
        );
    }
}
```

### 3.2 — Call Gemini REST API directly from .NET (no Python service needed)

If you prefer to skip the Python service and call Gemini directly:

```csharp
using System.Net.Http.Headers;
using System.Text.Json;

public class GeminiGymAnalyzer
{
    private const string Model = "gemini-2.0-flash";
    private readonly HttpClient _http;
    private readonly string _apiKey;

    // Paste the full system prompt from src/prompt.py here
    private const string SystemPrompt = """
        [paste ANALYSIS_PROMPT content here]
        """;

    public GeminiGymAnalyzer(HttpClient http, string apiKey)
    {
        _http = http;
        _apiKey = apiKey;
    }

    public async Task<JsonDocument> AnalyzeAsync(IEnumerable<string> imageUrls)
    {
        // Build inline image parts by fetching each URL
        var imageParts = new List<object>();
        foreach (var url in imageUrls)
        {
            var imageBytes = await _http.GetByteArrayAsync(url);
            var base64 = Convert.ToBase64String(imageBytes);
            imageParts.Add(new {
                inlineData = new { mimeType = "image/jpeg", data = base64 }
            });
        }

        var requestBody = new {
            contents = new[] {
                new {
                    parts = new object[] { new { text = SystemPrompt } }
                              .Concat(imageParts)
                              .Append(new { text = "Analyze these gym images and return JSON." })
                              .ToArray()
                }
            },
            generationConfig = new {
                responseMimeType = "application/json",
                temperature = 0.1
            }
        };

        var url2 = $"https://generativelanguage.googleapis.com/v1beta/models/{Model}:generateContent?key={_apiKey}";
        var json = JsonSerializer.Serialize(requestBody);
        var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

        var response = await _http.PostAsync(url2, content);
        response.EnsureSuccessStatusCode();

        var responseDoc = await JsonDocument.ParseAsync(
            await response.Content.ReadAsStreamAsync()
        );

        // Extract the text from candidates[0].content.parts[0].text
        var text = responseDoc
            .RootElement
            .GetProperty("candidates")[0]
            .GetProperty("content")
            .GetProperty("parts")[0]
            .GetProperty("text")
            .GetString();

        return JsonDocument.Parse(text!);
    }
}
```

---

## 4. Batch Processing

For large datasets, use the batch processor:

```bash
# Input: records.jsonl (one record per line)
# Format: {"gymId": "hotel-123", "imageUrls": ["url1", "url2"]}

python -m src.batch \
  --input  records.jsonl \
  --output results.jsonl \
  --failed failed.jsonl \
  --concurrency 10
```

**Input format** (`records.jsonl`):
```jsonl
{"gymId": "hotel-001", "imageUrls": ["https://...", "https://..."]}
{"gymId": "hotel-002", "imageUrls": ["https://..."]}
```

**Output format** (`results.jsonl`):
```jsonl
{"gymId": "hotel-001", "analysis": {...}, "processingTimeMs": 1842, "error": null}
{"gymId": "hotel-002", "analysis": {...}, "processingTimeMs": 1203, "error": null}
```

**Failed records** (`failed.jsonl`) — can be retried by passing as input again.

**Throughput estimates** (Gemini paid tier):

| Concurrency | Gyms/hour | 100k gyms |
|-------------|-----------|-----------|
| 10          | ~21,600   | ~4.6 hrs  |
| 50          | ~90,000   | ~1.1 hrs  |
| 100         | ~108,000  | ~55 min   |

**Cost estimate**: ~$0.001 per gym with gemini-2.0-flash = **~$100 for 100k gyms**.

The batch processor is **resumable** — if interrupted, re-run the same command and
it will skip already-processed `gymId`s.

---

## 5. Expected Latency

| Images per request | Typical latency |
|--------------------|-----------------|
| 1                  | 0.8 – 1.5 s     |
| 2–4                | 1.5 – 3.0 s     |
| 10–20              | 3.0 – 6.0 s     |

---

## 6. Retry & Error Handling

- The analyzer retries once automatically on malformed JSON from Gemini.
- For HTTP 429 (rate limit), the batch processor backs off exponentially.
- Recommended retry strategy for your .NET client:
  - 429: wait 5s, retry up to 3×
  - 500: wait 2s, retry up to 2×
  - 422: do not retry (bad input)
