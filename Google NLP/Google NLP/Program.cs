using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Google.Cloud.Language.V1;

using GoogleNLPConsole.Data;
using static Google.Cloud.Language.V1.AnnotateTextRequest.Types;

namespace GoogleNLPConsole
{
    public class Program
    {
        private static readonly string _connString = @"";
        private static readonly string _path = @"C:\output";

        public static void Main(string[] args)
        {
            // Before running through Google NLP api, I stored the board order text in a database in the board order table referenced in the db calls

            //Get order text and anazlyze
            var orders = NLPDatabase.GetBoardOrdersText(_connString);

            // Run through Google NLP api, save response in database in a Google table
            AnalyzeEverything(orders);

            ////Write sentences to output file
            //var responses = NLPDatabase.GetResponses( _connString);
            //responses = responses.Where(r => !r.Path.Contains("Sexual")).ToList();
            //ExtractSentences(responses, _spath);

            Console.Write("\nPress any key to continue...");
            Console.ReadKey();
        }

        public static void AnalyzeEverything(List<BoardOrder> orders)
        {
            var client = LanguageServiceClient.Create();
            var responses = new List<AnnotateTextResponse>();
            var truncatedOrders = new List<BoardOrder>();

            //Truncate orders with text exceeding the size limit.
            foreach (var order in orders.ToList())
            {
                if (order.Text.Length > 60000)
                {
                    var splitText = order.Text.SplitByLength(60000);
                    foreach (var text in splitText)
                    {
                        truncatedOrders.Add(new BoardOrder()
                        {
                            Path = order.Path,
                            Text = text,
                        });
                    }
                    orders.Remove(order);
                    continue;
                }
            }

            orders.AddRange(truncatedOrders);

            // Get text from each order and run through the api
            foreach (var order in orders)
            {
                var response = client.AnnotateText(new Document()
                {
                    //Removed new lines to get better output
                    Content = order.Text.Replace("\r\n", " "),
                    Type = Document.Types.Type.PlainText
                }, new Features()
                {
                    ExtractSyntax = true,
                    ExtractDocumentSentiment = true,
                    ExtractEntities = true,
                    ExtractEntitySentiment = true,
                    ClassifyText = true
                });

                order.SetResponse(response);
            }

            // save response
            NLPDatabase.SaveResponses(orders, _connString);

            return;
        }

        // Given output from Google NLP, store sentences in txt file. 
        public static void ExtractSentences(List<BoardOrder> responses, string path)
        {
            foreach (var response in responses)
            {
                var fileName = Path.GetFileName(response.Path);
                var fullPath = response.Path.Replace(_path, path);
                var filePath = response.Path.Replace(fileName, "").Replace(_path, path);
                Directory.CreateDirectory(filePath);
                using (var f = new StreamWriter(fullPath, true))
                {
                    foreach (var sentence in response.Response.Sentences)
                    {
                        f.WriteLine(sentence.Text.Content);
                    }
                }
            }
        }

        // Given output from Google NLP, store tokens in txt file. 
        public static void ExtractTokens(List<BoardOrder> responses, string path)
        {
            foreach (var response in responses)
            {
                var fileName = Path.GetFileName(response.Path);
                var fullPath = response.Path.Replace(_path, path);
                var filePath = response.Path.Replace(fileName, "").Replace(_path, path);
                Directory.CreateDirectory(filePath);
                using (var f = new StreamWriter(fullPath, true))
                {
                    foreach (var token in response.Response.Tokens)
                    {
                        f.WriteLine(token.Text.Content);
                    }
                }
            }
        }

        // Given output from Google NLP, store sentences in txt file. 
        public static void Sentences(List<BoardOrder> responses, string path)
        {
            using (var f = new StreamWriter(path))
            {
                foreach (var r in responses.Skip(5).Take(1))
                {
                    foreach (var s in r.Response.Sentences)
                    {
                        f.WriteLine(s.Text.Content.Replace("\r\n", ""));
                    }
                }
            }
        }

        // Given output from Google NLP, store entities in txt file. 
        public static void Entities(List<BoardOrder> responses, string path)
        {
            using (var f = new StreamWriter(path + "Entities.txt"))
            {
                foreach (var r in responses.Take(1))
                {
                    foreach (var e in r.Response.Entities)
                        f.WriteLine(e.Name);
                }
            }
        }

    }
}
