namespace Hockey.Shared.Dto
{
    public class VideoInfoDto
    {
        public string FileName { get; set; }
        public string FirstTeamName { get; set; }
        public string SecondTeamName { get; set; }

        public byte[] FirstTeamColor { get; set; }
        public byte[] SecondTeamColor { get; set; }
    }
}
