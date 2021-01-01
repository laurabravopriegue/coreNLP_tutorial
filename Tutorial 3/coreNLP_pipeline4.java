package edu.stanford.nlp.examples;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;

import edu.stanford.nlp.trees.*;
import java.util.*;
import edu.stanford.nlp.sentiment.*;
import edu.stanford.nlp.io.*;
import java.io.FileNotFoundException;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import org.ejml.simple.SimpleMatrix;
import java.io.BufferedReader;

public class predictSentiment {

  public static void main(String[] args) throws IOException {

    // set up pipeline properties
    Properties props = new Properties();
    // set the list of annotators to run
    props.setProperty("annotators","tokenize,ssplit,parse,sentiment");
    // build pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    // Get dataset name from argument one
    String dataset = args[1];
    //select output document
    System.out.println("Create and open output document...");
    File file = new File("predictions/predictions_" + dataset + ".txt");
    //create the file if it doesn't exist
    if(!file.exists()){
      file.createNewFile();}

    PrintWriter out = new PrintWriter(file);

    //print column names on the output document
    out.println("review_id,sent_id,sentence,sentiment,veryneg,neg,neu,pos,verypos");

    // get file to read from argument zero
    BufferedReader br = null;
    String filename = args[0];

    br = new BufferedReader ( new java.io.FileReader (filename)  ) ;
    String readString = null;


    int review_id = 0;

    String x = null;
    // Each new line is a review
    while  (( readString = br.readLine())  != null)   {

      x = readString;

      int sent_id = 0;

      System.out.println("Processing review " + review_id);

      // create a document object
      CoreDocument document = pipeline.processToCoreDocument(x);
      //annotate document
      pipeline.annotate(document);


      //get list of sentences
      List <CoreSentence> sentences = document.sentences();

      //iterate through sentences

      for (CoreSentence sentence : sentences) {

        System.out.println("Sentence " + sent_id);

        sent_id += 1;

        Tree tree = sentence.sentimentTree();
          //get overall score
        int sentimentScore =  RNNCoreAnnotations.getPredictedClass(tree);


        SimpleMatrix simpleMatrix = RNNCoreAnnotations.getPredictions(tree);

          //Gets prob for each sentiment using the elements of the sentimet matrix
                 float veryneg = (float)Math.round((simpleMatrix.get(0)*100d));
                 float neg = (float)Math.round((simpleMatrix.get(1)*100d));
                 float neutral = (float)Math.round((simpleMatrix.get(2)*100d));
                 float pos = (float)Math.round((simpleMatrix.get(3)*100d));
                 float verypos = (float)Math.round((simpleMatrix.get(4)*100d));


          //printing data to the output document
          out.println(review_id + "," + sent_id + "," + sentence + "," + sentimentScore + "," + veryneg + "," + neg + "," + neutral + "," + pos + "," + verypos);


        }

        review_id += 1;
      }

      // Close reader and writer

      br.close();
      out.close () ;

      System.out.println(" Done!");

      }

    }
