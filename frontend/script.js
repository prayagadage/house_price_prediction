document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating...';
    btn.disabled = true;

    // Gather data
    const data = {
        BHK: parseInt(document.getElementById('bhk').value),
        Size: parseInt(document.getElementById('size').value),
        Bathroom: parseInt(document.getElementById('bathroom').value),
        current_floor: parseInt(document.getElementById('current_floor').value),
        total_floors: parseInt(document.getElementById('total_floors').value),
        Area_Type: document.getElementById('area_type').value,
        Area_Locality: document.getElementById('locality').value || "Other",
        City: document.getElementById('city').value,
        Furnishing_Status: document.getElementById('furnishing').value,
        Tenant_Preferred: document.getElementById('tenant').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        // Show result
        const resultArea = document.getElementById('resultArea');
        const priceValue = document.getElementById('priceValue');

        resultArea.classList.remove('result-hidden');

        // Animate counting
        animateValue(priceValue, 0, result.predicted_rent, 1000);

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Ensure the backend is running.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        obj.innerHTML = `₹ ${value.toLocaleString('en-IN')}`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
