# Frontend-Backend Integration Debugging Guide

## Quick Test Steps

### 1. Test Backend Health Check (Direct)
Open this URL in your browser:
```
https://ai-visa-prediction-production.up.railway.app/health
```
**Expected Response:**
```json
{"status": "ok", "message": "VisaAI Backend API is running"}
```

---

### 2. Test Frontend Access
Open your frontend:
```
https://frontend-b8kry1gf1-gulshan-kumars-projects-9ec96ed7.vercel.app/
```

---

### 3. Check Browser Console Logs
1. Open the frontend URL
2. Press **F12** to open Developer Tools
3. Go to **Console** tab
4. Fill in the form:
   - Country: India
   - Visa Type: Student
   - Application Date: 2024-01-15
5. Click **Estimate Processing Time**
6. Check console output for:

**Should see logs like:**
```
Using default backend URL: https://ai-visa-prediction-production.up.railway.app
Form submitted with: {country: 'India', visa_type: 'Student', application_date: '2024-01-15'}
Calling backend at: https://ai-visa-prediction-production.up.railway.app
Request data: {country: 'India', visa_type: 'Student', application_date: '2024-01-15', processing_office: null}
Response status: 200
Response data: {success: true, country: 'India', visa_type: 'Student', application_date: '2024-01-15', estimated_days: 45.3}
Prediction successful: 45.3
```

---

### 4. Common Issues & Fixes

#### Issue: Response status is not 200
**Possible causes:**
- Backend server crashed or restarted
- Model files missing on Railway
- Memory/timeout issues

**Fix:** Check Railway dashboard and redeploy

#### Issue: CORS error
**Possible causes:**
- Frontend domain not allowed by backend
- CORS headers missing

**Fix:** Already enabled in api.py with `CORS(app)`

#### Issue: JSON parsing error
**Possible causes:**
- Backend returning HTML error page instead of JSON
- Connection timeout

**Fix:** Check backend logs on Railway

#### Issue: Backend URL undefined
**Possible causes:**
- localStorage not initialized
- config.html not visited

**Fix:** Visit `/config.html` page to initialize backend URL

---

### 5. Manual API Test (Curl/PowerShell)

**PowerShell Test:**
```powershell
$body = @{
    country = "India"
    visa_type = "Student"
    application_date = "2024-01-15"
    processing_office = $null
} | ConvertTo-Json

$response = Invoke-WebRequest `
  -Uri "https://ai-visa-prediction-production.up.railway.app/predict" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body `
  -TimeoutSec 30

$response.Content | ConvertFrom-Json | Format-Table
```

---

### 6. Configuration Page Test

Visit configuration page to ensure backend URL is saved:
```
https://frontend-b8kry1gf1-gulshan-kumars-projects-9ec96ed7.vercel.app/config.html
```

**Actions:**
1. Verify backend URL field shows: `https://ai-visa-prediction-production.up.railway.app`
2. Click "Test Connection" button
3. Should see: ✅ Connected! Backend is running

---

## Expected Success Flow

1. ✅ Frontend loads
2. ✅ Browser loads localStorage backend URL
3. ✅ User fills form and submits
4. ✅ JavaScript fetch() calls backend /predict endpoint
5. ✅ Backend loads model and computes prediction
6. ✅ Backend returns JSON with estimated_days
7. ✅ Frontend shows result page with prediction

---

## If Still Getting Error

**Copy-paste the complete console output and share:**
- Full error message from alert
- Console logs showing request/response details
- Response status code
- Any error messages from backend

This will help identify exactly where the integration is failing.
