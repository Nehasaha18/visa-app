// Backend API Configuration
let BACKEND_API_URL = 'https://visa-backend-maiz.onrender.com';

// Check if user has configured a custom backend URL
const savedUrl = localStorage.getItem('backendUrl');
if (savedUrl) {
  BACKEND_API_URL = savedUrl;
  console.log('Using saved backend URL:', BACKEND_API_URL);
} else {
  console.log('Using default backend URL:', BACKEND_API_URL);
}

// Update backend URL from frontend config
function setBackendUrl(url) {
  localStorage.setItem('backendUrl', url);
  BACKEND_API_URL = url;
  console.log('Backend URL updated to:', url);
}

// API call function
async function predictVisa(country, visa_type, application_date, processing_office = null) {
  try {
    console.log('Calling backend at:', BACKEND_API_URL);
    console.log('Request data:', { country, visa_type, application_date, processing_office });
    
    const response = await fetch(`${BACKEND_API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        country,
        visa_type,
        application_date,
        processing_office
      })
    });

    console.log('Response status:', response.status);
    const data = await response.json();
    console.log('Response data:', data);

    if (!response.ok) {
      throw new Error(`Backend error (${response.status}): ${data.error || 'Unknown error'}`);
    }

    if (!data.success) {
      throw new Error(data.error || 'Prediction failed');
    }

    return data;
  } catch (error) {
    console.error('Prediction error:', error);
    return { success: false, error: error.message };
  }
}

// Form submission handler
document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('estimateForm');
  if (form) {
    form.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const country = form.querySelector('[name="country"]').value;
      const visa_type = form.querySelector('[name="visa_type"]').value;
      const application_date = form.querySelector('[name="application_date"]').value;
      
      console.log('Form submitted with:', { country, visa_type, application_date });
      
      if (!country || !visa_type || !application_date) {
        alert('Please fill in all fields');
        return;
      }

      const result = await predictVisa(country, visa_type, application_date);
      
      if (result.success) {
        console.log('Prediction successful:', result.estimated_days);
        showPredictionResult(result.estimated_days, country, visa_type, application_date);
      } else {
        console.error('Prediction failed:', result.error);
        alert(`Error: ${result.error}`);
      }
    });
  }
});

// Show prediction result
function showPredictionResult(days, country, visaType, date) {
  const html = `<html>
    <body style="font-family:Inter, Poppins, sans-serif;background:#07104a;color:#eaf0ff;display:flex;align-items:center;justify-content:center;height:100vh">
      <div style="background:rgba(255,255,255,0.02);padding:40px;border-radius:12px;box-shadow:0 20px 40px rgba(0,0,0,0.6);text-align:center;max-width:500px">
        <h2 style="font-size:28px;margin-bottom:20px">✅ Visa Processing Time Estimate</h2>
        <div style="background:rgba(10,97,255,0.2);padding:30px;border-radius:8px;margin:20px 0">
          <div style="font-size:48px;font-weight:bold;color:#1ec5ff">${days}</div>
          <div style="font-size:18px;color:#aaa;margin-top:10px">Estimated Days</div>
        </div>
        <div style="text-align:left;margin:20px 0;padding:20px;background:rgba(255,255,255,0.05);border-radius:8px">
          <p><strong>Country:</strong> ${country}</p>
          <p><strong>Visa Type:</strong> ${visaType}</p>
          <p><strong>Application Date:</strong> ${date}</p>
        </div>
        <p><a href="/" style="color:#0a61ff;text-decoration:none">← Go Back</a></p>
      </div>
    </body>
  </html>`;
  
  document.open();
  document.write(html);
  document.close();
}
