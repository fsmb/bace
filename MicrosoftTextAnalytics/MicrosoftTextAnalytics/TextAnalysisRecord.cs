using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Azure.CognitiveServices.Language.TextAnalytics.Models;
using Newtonsoft.Json;

namespace MicrosoftTextAnalytics
{
    public class TextAnalysisRecord
    {
        public TextAnalyticsRequest RequestInfo { get; set; }

        public IList<EntityRecord> Entities { get; set; }
        public IEnumerable<string> EntityErrors { get; set; }

        public IList<string> KeyPhrases { get; set; }
        public IEnumerable<string> KeyPhraseErrors { get; set; }
        public double? Sentiment { get; set; }
        public IEnumerable<string> SentimentErrors { get; set; }

        public string Fid { get; set; }
        [JsonIgnore]
        public string CleanDocText { get; set; }
        public int OrderNumber { get; set; }

        public Category? Category { get; set; }
        //public string FileName { get; set; }

    }


    public enum Category
    {
        Sex = 1,
        Drugs = 2,
        Other = 99
    }

    public static class TextAnalysisRecordExtensions
    {
        public static void SaveRecordJsonFile(this TextAnalysisRecord source, string pathToSaveTo )
        {
            Directory.CreateDirectory(Path.GetDirectoryName(pathToSaveTo));
            try
            {
                using (StreamWriter file = File.CreateText(pathToSaveTo))
                {
                    JsonSerializer serializer = new JsonSerializer();
                    serializer.Formatting = Formatting.Indented;
                    serializer.Serialize(file, source);
                }
            } catch (Exception e)
            {
            }
        }
    }
}
