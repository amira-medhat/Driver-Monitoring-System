$(document).ready(function () {

    let isMonitoring = false;  // Global flag to check if the assistant is in monitoring mode

    // --- Text animation for assistant messages ---
    $('.text').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeIn",
        },
        out: {
            effect: "fadeOutUp",
        },
    });

    // --- SiriWave configuration ---
    var siriWave = new SiriWave({
        container: document.getElementById("siri-container"),
        width: 800,
        height: 200,
        style: "ios9",
        color: "#fff",
        speed: 0.2,
        amplitude: 1,
        autostart: true
    });

    // --- Siri Message Animation ---
    $('.siri-message').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeInUp",
            sync: true,
        },
        out: {
            effect: "fadeOutUp",
            sync: true,
        },
    });

    // --- Mic Button Click Event ---
    $("#MicBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden", true);
        $("#SiriWave").attr("hidden", false);
        eel.set_mic_pressed();  // Notify Python that mic button was pressed
    });

    // --- Settings Button Click Event ---
    $("#SettingsBtn").click(function () {
        $("#SettingsWindow").fadeToggle(); // Toggle Settings window
    });

    // --- Close Settings Button Click Event ---
    $("#CloseSettings").click(function () {
        $("#SettingsWindow").fadeOut();
    });

    // --- GPS Button Click Event ---
    $("#GpsBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.OpenGps("gps");
    });

    // --- Monitor ON/OFF Buttons ---

    // Function to update UI buttons based on backend state
    function updateMonitorButtons() {
        eel.get_monitor_mode()(function (mode) {
            if (mode === "on") {
                $("#MonitorOnBtn").addClass("selected-option");
                $("#MonitorOffBtn").removeClass("selected-option");
            } else {
                $("#MonitorOffBtn").addClass("selected-option");
                $("#MonitorOnBtn").removeClass("selected-option");
            }
        });
    }

    // Monitor ON button click
    $("#MonitorOnBtn").click(function () {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.set_monitoring_state("monitoring");  // New backend function
        updateMonitorButtons();
    });

    // Monitor OFF button click
    $("#MonitorOffBtn").click(function () {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.set_monitoring_state("idle");  // New backend function
        updateMonitorButtons();
    });

    // --- Initialize Monitor Button state on page load ---
    updateMonitorButtons();
    setInterval(updateMonitorButtons, 2000);  // Poll every 2 seconds

});

// Instructions Button Click
$("#InstructionsBtn").click(function () {
    $("#SettingsWindow").fadeOut();        // Hide Settings window
    $("#InstructionsWindow").fadeToggle(); // Toggle Instructions window
});

// Close Instructions Window
$("#CloseInstructionsBtn").click(function () {
    $("#InstructionsWindow").fadeOut();
});

eel.expose(selectMonitorOnButton);
function selectMonitorOnButton(flag) {
    if (flag == true) {
        $("#MonitorOnBtn").addClass("selected-option");
        $("#MonitorOffBtn").removeClass("selected-option");
    } else {
        $("#MonitorOffBtn").addClass("selected-option");
        $("#MonitorOnBtn").removeClass("selected-option");
    }
}
