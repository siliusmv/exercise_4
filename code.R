
## ---- knn

# Euclidian distance function
eucDist <- function(all_points, point){
  
  res <- sweep(all_points, 2, point, "-")
  res <- res^2
  res <- apply(res, 1, sum)
  res <- sqrt(res)
  
  return(res)
}


# Find k nearest neighbours
findKnn <- function(all_points, point, distFunc, k, values){
  
  distances <- distFunc(all_points, point)
  
  ord <- order(distances)
  
  res <- list(distances = distances[1:k],
              values = values[ord][1:k])
  
  return(res)
  
}


# Classification by voting
vote <- function(closest){
  
  t <- table(closest$values)
  
  or <- order(t, decreasing = TRUE)
  
  if(t[or[1]] == t[or[2]]){
    # return("You need to fix voting when two or more classes 
           # are equally well represented")
    equal_votes <- t[which(t == t[or[1]])]
    
    for(i in 1:k){
      p <- which(names(equal_votes) == as.character(closest$values[i])) 
      if(p){
        return(names(equal_votes)[p])
      }
    }
  }
  
  return(names(t)[or[1]])
}


knn <- function(k, data, point, type){
  
  all_points <- as.matrix(data[, -dim(data)[2]])
  values <- data[, dim(data)[2]]
  
  closest <- findKnn(all_points = all_points,
                     point = point,
                     k = k,
                     values = values,
                     distFunc = eucDist)
  
  if(type == "reg"){
    return(mean(closest$values))
  } else if(type == "class"){
    return(vote(closest))
  }
  
  return("This type is not accepted")
}


