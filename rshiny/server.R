server <- function(input, output) {
  filtered_data <- reactive({
    # req(input$search)  # Action button controls the operation
    subset(df, 
           ProductType == input$productType & 
             Weight == input$weight & 
             HalalStatus == input$halal & 
             HealthStatus == input$healthy)
  })
  
  output$imageGallery <- renderUI({
    selected <- filtered_data()
    if (nrow(selected) > 0) {
      image_tags <- lapply(1:nrow(selected), function(i) {
        div(class = "image-container", 
            tags$img(src = #paste0("/Users/liupeng/Documents/GitHub/object_detection_using_tensorflow/images_combined/",
                       paste0("all_images/",
                                  selected$filepath[i]), 
                     alt = selected$label[i], 
                     style = "max-width: 100%; height: auto;")
        )
      })
      div(class = "row", do.call(tagList, image_tags))
    }
  })
  
  output$filtered_tbl = renderTable({
    filtered_data()["filepath"]
  })
  
  
  # output$testImage <- renderUI({
  #   tags$img(src = "data_edit/IMG_20230428_123528_jpg.rf.5687b7b914f6d9aa98cadf060d1e3b00.jpg", 
  #            alt = "Test Image")
  # })
  
}
