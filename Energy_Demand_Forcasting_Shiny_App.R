library(shiny)
library(later)
library(readxl)

ui <- fluidPage(
  tags$head(
    tags$style(HTML("body { background-color: #f8f9fa; font-family: 'Helvetica Neue', sans-serif; }
                    .well { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; padding: 15px; }
                    h3 { color: #2c3e50; } "))
  ),
  
  titlePanel("\U1F4CA Forecasting July Energy Use in Homes: A +5\u00b0C Scenario"),
  
  div(class = "well",
      h3("\U1F4D6 Project Overview"),
      p("This project aimed to evaluate how residential energy demand would change with a projected +5°C increase in July temperatures. Using historical data from 5000 homes in the Carolinas, we trained multiple predictive models — with Random Forest demonstrating the best performance (MSE: ~0.375).

We found that temperature, humidity, cooling setpoints, insulation, and appliance usage were among the top drivers of energy usage. The models predicted 14.5% increase in total energy consumption per home in a warmer July, with significant shifts in hourly demand peaks.

These insights suggest that climate resilience planning, such as upgrading HVAC systems, shifting load through smart devices, and optimizing insulation, is critical. Additionally, targeted behavioral changes and energy-aware appliance choices can make residential energy systems more efficient and climate-ready.")
  ),
  
  sidebarLayout(
    sidebarPanel(
      actionButton("show_data", "📄 View Sample Data", class = "toggle-btn"),
      selectInput("model_choice", "Compare Model Results:",
                  choices = c("-- Select --", "Linear Regression (LM)", "Random Forest (RF)")),
      selectInput("insight_choice", "Actionable Insights:",
                  choices = c("-- Select --",
                              "Energy Demand Rises with +5\u00b0C",
                              "Top 5 Device Energy Consumption",
                              "Hourly Usage by Device",
                              "Device Contribution to Energy"))
    ),
    
    mainPanel(
      div(class = "well",
          uiOutput("dynamic_content")
      )
    )
  )
)

server <- function(input, output, session) {
  show_data_flag <- reactiveVal(FALSE)
  dropdown_active <- reactiveVal(FALSE)
  excel_data <- reactiveVal(NULL)  # to store Excel content
  
  observeEvent(input$show_data, {
    # Clear dropdowns if selected
    if (input$model_choice != "-- Select --") {
      updateSelectInput(session, "model_choice", selected = "-- Select --")
      dropdown_active(TRUE)
    }
    if (input$insight_choice != "-- Select --") {
      updateSelectInput(session, "insight_choice", selected = "-- Select --")
      dropdown_active(TRUE)
    }
    
    # Read Excel after slight delay
    later::later(function() {
      file_path <- "model_data1.xlsx"  # use forward slashes!
      if (file.exists(file_path)) {
        df <- tryCatch(read_excel(file_path), error = function(e) {
          cat("❌ Error reading Excel:", e$message, "\n")
          NULL
        })
        excel_data(df)
      } else {
        cat("❌ File not found:", file_path, "\n")
        excel_data(NULL)
      }
      show_data_flag(TRUE)
      dropdown_active(FALSE)
    }, delay = 0.2)
  })
  
  observeEvent(input$model_choice, {
    if (input$model_choice != "-- Select --") {
      updateSelectInput(session, "insight_choice", selected = "-- Select --")
      show_data_flag(FALSE)
    }
  })
  
  observeEvent(input$insight_choice, {
    if (input$insight_choice != "-- Select --") {
      updateSelectInput(session, "model_choice", selected = "-- Select --")
      show_data_flag(FALSE)
    }
  })
  
  output$dynamic_content <- renderUI({
    list_items <- list()
    
    if (!dropdown_active() && input$model_choice != "-- Select --") {
      list_items <- append(list_items, list(
        h3("\U1F4A1 Model Summary:"),
        verbatimTextOutput("model_summary"),
        imageOutput("model_image")
      ))
    } else if (!dropdown_active() && input$insight_choice != "-- Select --") {
      list_items <- append(list_items, list(
        h3("\U1F4A1 Insight Summary:"),
        verbatimTextOutput("insight_summary"),
        imageOutput("insight_image")
      ))
    }
    
    if (show_data_flag()) {
      list_items <- append(list_items, list(
        h3("✅ Data Load Confirmation"),
        tableOutput("data_preview")
      ))
    }
    
    do.call(tagList, list_items)
  })
  
  output$data_status <- renderText({
    df <- excel_data()
    if (is.null(df)) {
      "file does not exits"
    } else {
      paste("✅ Successfully loaded", nrow(df), "rows and", ncol(df), "columns from Excel.")
    }
  })
  
  output$data_preview <- renderTable({
    df <- excel_data()
    if (!is.null(df)) head(df, 10)
  })
  
  output$model_summary <- renderText({
    switch(input$model_choice,
           "Linear Regression (LM)" = "Linear Regression MSE: ~0.566. Easier to interpret but less accurate.
How to Interpret:
Each point compares actual vs predicted energy. A perfect model will have all points near the red diagonal. 
Random Forest performs better for non-linear trends like cooling demand.",
           "Random Forest (RF)" = "Random Forest MSE: ~0.396. Handles complex patterns in energy use.
How to Interpret:
Each point compares actual vs predicted energy. A perfect model will have all points near the red diagonal. 
Random Forest performs better for non-linear trends like cooling demand.")
  })
  
  output$insight_summary <- renderText({
    switch(input$insight_choice,
           "Energy Demand Rises with +5\u00b0C" = "Energy may rise ~14.5% in warmer July.",
           "Top 5 Device Energy Consumption" = "Cooling and plug loads lead total energy use.",
           "Hourly Usage by Device" = "Evening peaks from cooling and plug load devices.",
           "Device Contribution to Energy" = "Pie chart shows cooling systems are the largest contributor.")
  })
  
  output$model_image <- renderImage({
    filename <- switch(input$model_choice,
                       "Linear Regression (LM)" = "lm.png",
                       "Random Forest (RF)" = "rf.png")
    list(src = filename, contentType = 'image/png', width = 600)
  }, deleteFile = FALSE)
  
  output$insight_image <- renderImage({
    filename <- switch(input$insight_choice,
                       "Energy Demand Rises with +5\u00b0C" = "plusperhouse.png",
                       "Top 5 Device Energy Consumption" = "top5.png",
                       "Hourly Usage by Device" = "hourperdevice.png",
                       "Device Contribution to Energy" = "devicecontribution.png")
    list(src = filename, contentType = 'image/png', width = 600)
  }, deleteFile = FALSE)
}

shinyApp(ui, server)
