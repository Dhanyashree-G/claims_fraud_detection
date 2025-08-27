document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic
    const tabs = document.querySelectorAll(".tab-button");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            // Remove active class from all buttons and hide all content
            document.querySelectorAll(".tab-button").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            // Add active class to clicked button and show corresponding content
            tab.classList.add("active");
            const tabName = tab.dataset.tab;
            document.getElementById(tabName).classList.add("active");
        });
    });

    // Set initial active tab (if not already set by server-side rendering)
    const initialActiveTab = document.querySelector(".tab-button.active");
    if (initialActiveTab) {
        const initialTabContent = document.getElementById(initialActiveTab.dataset.tab);
        if (initialTabContent) {
            initialTabContent.classList.add("active");
        }
    } else {
        // Fallback: activate the first tab if none is active
        if (tabs.length > 0) {
            tabs[0].classList.add("active");
            document.getElementById(tabs[0].dataset.tab).classList.add("active");
        }
    }


    // Show loading overlay for provider upload
    const predictForm = document.getElementById("predict-form");
    if (predictForm) {
        predictForm.addEventListener("submit", () => {
            const overlay = document.getElementById("loading-overlay");
            if (overlay) {
                overlay.style.display = "flex"; // Use flex to center content
            }
        });
    }

    // User menu dropdown toggle
    const userMenuToggle = document.querySelector('.user-menu-toggle');
    const userMenuDropdown = document.querySelector('.user-menu-dropdown');

    if (userMenuToggle && userMenuDropdown) {
        userMenuToggle.addEventListener('click', (event) => {
            event.stopPropagation(); // Prevent click from immediately closing
            userMenuDropdown.classList.toggle('show');
        });

        // Close dropdown if clicked outside
        document.addEventListener('click', (event) => {
            if (!userMenuToggle.contains(event.target) && !userMenuDropdown.contains(event.target)) {
                userMenuDropdown.classList.remove('show');
            }
        });
    }

    // Flash message auto-hide (optional)
    const flashMessages = document.querySelectorAll('.flash');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            msg.style.transition = 'opacity 0.5s ease-out';
            setTimeout(() => msg.remove(), 500); // Remove after transition
        }, 5000); // Hide after 5 seconds
    });
});
