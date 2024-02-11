server <- function(input, output) {
  image_path <- reactive({
    input$input_image_upload$datapath
    })

  prediction <- reactive({
    req(image_path(), input$min_score_threshold)
    
    if(is.null(image_path())){return(NULL)}
    pred = predict_class(img_path=image_path(),
                         min_score_thresh=input$min_score_threshold)
    # print(pred)
    pred = convert_scoring_rst(pred)
    pred
  })
  
  output$text <- renderTable({
    req(image_path(), input$min_score_threshold)
    
    prediction()
  })
  
  # Show uploaded image
  output$output_image <- renderImage({
    req(image_path())
    
    outfile <- image_path()
    contentType <- "image/jpeg"
    list(src = outfile,
         contentType=contentType,
         width = 400)
  }, deleteFile = FALSE)
  
  # Show selected image
  output$output_image_selected <- renderImage({
    req(input$input_image_select)
    
    outfile <- paste0('images/',input$input_image_select)
    print(outfile)
    contentType <- "image/jpeg"
    list(src = outfile,
         contentType=contentType,
         width = 400)
  }, deleteFile = FALSE)
  
  output$output_image2 <- renderImage({
    req(image_path(), input$min_score_threshold)
    
    outfile <- "tmp_output/img.jpg"
    contentType <- "jpg"
    list(src = outfile,
         contentType=contentType,
         width = 400)
  }, deleteFile = FALSE)
  
  # show all product categories
  output$all_cats <- renderText({
    ALL_CATS
  })
}